import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import os 
import jwt
import vertexai
import base64
from wordcloud import WordCloud
from PIL import Image
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel

PROJECT = 'keboola-ai'
LOCATION = 'us-central1'
MODEL_NAME = 'gemini-1.5-pro-preview-0409'

CREDENTIALS = service_account.Credentials.from_service_account_info(
    jwt.decode(st.secrets['encoded_token'], 'keboola', algorithms=['HS256'])
)

def color_for_value(value):
    if value < -0.2:
        return '#EA4335'
    elif -0.2 <= value <= 0.2:
        return '#FBBC05' 
    else:
        return '#34A853' 

def categorize_sentiment(score):
    if score < -0.2:
        return 'Negative'
    elif -0.2 <= score <= 0.2:
        return 'Neutral'
    else:
        return 'Positive'

def sentiment_color(sentiment):
    if sentiment == "Positive":
        return "color: #34A853"
    if sentiment == "Neutral":
        return "color: #FBBC05"
    else:
        return "color: #EA4335"

# Gemini 
def generate(content):
    try:
        vertexai.init(project=PROJECT, location=LOCATION, credentials=CREDENTIALS)
        model = GenerativeModel(MODEL_NAME)

        config = {
            'max_output_tokens': 8192,
            'temperature': 1,
            'top_p': 0.95,
        }
        
        responses = model.generate_content(
            contents=content,
            generation_config=config,
            stream=True,
        )
        return "".join(response.text for response in responses)
    except Exception as e:
        st.error(f"An error occurred during content generation: {e}")
        return None

st.set_page_config(layout='wide')

image_path = os.path.dirname(os.path.abspath(__file__))

keboola_logo = image_path+"/static/keboola_logo.png"
logo_html = f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{base64.b64encode(open(keboola_logo, "rb").read()).decode()}" style="width: 200px; margin-bottom: 10px;"></div>'
st.markdown(f"{logo_html}", unsafe_allow_html=True)

st.title('London Eye Reviews Sentiment Analysis')

data_path = '/data/in/tables/reviews_sentiment_final_gemini.csv'
keywords_path = '/data/in/tables/reviews_keywords_final_gemini.csv'

@st.cache_data
def get_data():
    data = pd.read_csv(data_path, parse_dates=['parsed_date'])
    keywords = pd.read_csv(keywords_path)
    return data, keywords

data, keywords = get_data()

data['sentiment_category'] = data['sentiment'].apply(categorize_sentiment)
data['parsed_date'] = pd.to_datetime(data['parsed_date'])
data['date'] = data['parsed_date'].dt.date
data['is_widget'] = False

keywords['parsed_date'] = pd.to_datetime(keywords['parsed_date'])
keywords['date'] = keywords['parsed_date'].dt.date

values_to_exclude = ['London', 'London Eye']
keywords_filtered = keywords[~keywords['keywords'].isin(values_to_exclude)]

st.markdown("<br>__Filters__", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3, gap='medium')
st.markdown("<br>", unsafe_allow_html=True)    

with col1:
    min_score, max_score = st.slider(
                'Select a range for the sentiment score:',
                min_value=-1.0, max_value=1.0, value=(-1.0, 1.0),
                key="sentiment_slider"
            )

with col2:
    source_choices = ['All'] + data['reviewSource'].unique().tolist()
    selected_sources = st.selectbox('Filter by review source:', source_choices)

with col3:
    if not data.empty:
        min_date = data['parsed_date'].min()
        max_date = data['parsed_date'].max()
        default_date_range = (min_date, max_date)
    else:
        min_date, max_date = None, None
        default_date_range = ()

    date_range = st.date_input("Select a date range:", default_date_range, min_value=min_date, max_value=max_date)
if date_range:
    try:
        if len(date_range) == 2:
            start_date, end_date = date_range
            data = data[(data['parsed_date'] >= pd.to_datetime(start_date)) & (data['parsed_date'] <= pd.to_datetime(end_date))]
            keywords_filtered = keywords_filtered[(keywords_filtered['parsed_date'] >= pd.to_datetime(start_date)) & (keywords_filtered['parsed_date'] <= pd.to_datetime(end_date))]
    except Exception as e:
        st.info("Please select both start and end dates.")

# Apply Filters
if selected_sources == 'All':
    filtered_data = data[(data['sentiment'] >= min_score) & (data['sentiment'] <= max_score)]
    keywords_filtered = keywords_filtered[(keywords_filtered['sentiment'] >= min_score) & (keywords_filtered['sentiment'] <= max_score)]
else:
    filtered_data = data[(data['sentiment'] >= min_score) & (data['sentiment'] <= max_score) & (data['reviewSource'] == selected_sources)]
    keywords_filtered = keywords_filtered[(keywords_filtered['sentiment'] >= min_score) & (keywords_filtered['sentiment'] <= max_score) & (keywords_filtered['reviewSource'] == selected_sources)]

#unique_sentiment_scores = filtered_data['sentiment'].unique()
#keywords_filtered = keywords_filtered[keywords_filtered['sentiment'].isin(unique_sentiment_scores)]


col1, col2, col3 = st.columns(3, gap='medium')
with col1:
    filtered_data['color'] = filtered_data['sentiment'].apply(color_for_value)

    fig = px.histogram(
        filtered_data,
        x='sentiment',
        nbins=21,  
        title='Sentiment Score Distribution',
        color='color',
        color_discrete_map='identity'  
    )

    fig.update_layout(bargap=0.1, xaxis_title='Sentiment Score', yaxis_title='Count') 
    st.plotly_chart(fig, use_container_width=True)

with col2: 
    keyword_counts = keywords_filtered.groupby('keywords')['counts'].sum().reset_index()
    top_keywords = keyword_counts.sort_values(by='counts', ascending=True).tail(10)
    #st.write(keywords_filtered)
    #top_keywords = keywords_filtered.sort_values(by='counts', ascending=True).tail(10)
	
    fig = px.bar(top_keywords, x='counts', y='keywords', orientation='h', title='Top 10 Keywords by Count', color_discrete_sequence=['#4285F4'])
    fig.update_layout(xaxis_title='Count', yaxis_title='Keywords')

    st.plotly_chart(fig, use_container_width=True)

with col3:
    review_source_counts  = filtered_data['reviewSource'].value_counts().reset_index()
    review_source_counts .columns = ['reviewSource', 'count']
    top_10_industries = review_source_counts .head(10)
    count = top_10_industries.shape[0]
    colors = ['#D4D4D4', '#939393']
    
    fig = px.pie(
        top_10_industries, 
        names='reviewSource', 
        values='count', 
        title=f'Distribution of Reviews',
        color_discrete_sequence=colors
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([3,2])

# Show table
with col1:
    st.markdown("__Data__")
    sorted_data = filtered_data.sort_values(by='parsed_date', ascending=False)
    selected_data = st.data_editor(sorted_data[['is_widget',
                                'sentiment_category',
                                'text_in_english',                            
                                'rating',
                                'reviewSource', 
                                'date',
                                'url']].style.applymap(
            sentiment_color, subset=["sentiment_category"]
        ), 
                column_config={'is_widget': 'Select',
                                'sentiment_category': 'Sentiment Category',
                                'text_in_english': 'Text',
                                'rating': 'Rating',
                                'reviewSource': 'Review Source',
                                'date': 'Date',
                                'url': st.column_config.LinkColumn('URL')
                                }, height=500,
                 disabled=['sentiment_category',
                           'text_in_english',                            
                           'rating',
                           'reviewSource', 
                           'date',
                           'url'],
                use_container_width=True, hide_index=True)

@st.cache_data
def generate_wordcloud(word_freq, mask_image_path):
    colormap = mcolors.ListedColormap(['#4285F4', '#34A853', '#FBBC05', '#EA4335'])
    mask_image = np.array(Image.open(mask_image_path))

    # Ensure mask_image is of type uint8
    if mask_image.dtype != np.uint8:
        mask_image = mask_image.astype(np.uint8)

    wordcloud = WordCloud(width=500, height=500, background_color=None, 
                          colormap=colormap, mask=mask_image,
                          mode='RGBA').generate_from_frequencies(word_freq)
    wordcloud_array = wordcloud.to_array()

    # Ensure the wordcloud_array is of type uint8
    if wordcloud_array.dtype != np.uint8:
        wordcloud_array = wordcloud_array.astype(np.uint8)
    
    return wordcloud_array
    
summary = keywords_filtered.groupby('keywords')['counts'].sum().reset_index()
#filtered_summary = summary[summary['counts'] > 10]
#filtered_summary = keywords_filtered[keywords_filtered['counts'] > 10]
#word_freq = dict(zip(filtered_summary['keywords'], filtered_summary['counts']))
word_freq = dict(zip(summary['keywords'], summary['counts']))


# Wordcloud
with col2:
    st.markdown("__Word Eye__")
    # Generate the word cloud
    if word_freq:    
        wordcloud = generate_wordcloud(word_freq, image_path + "/static/london_eye_wc.png")
        fig, ax = plt.subplots(figsize=(10, 5), frameon=False)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')    
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("No keywords found to generate the word cloud.")
        
# Gemini response
keboola_gemini = image_path + "/static/keboola_gemini.png"
gemini_html = f'<div style="display: flex; justify-content: flex-end;"><img src="data:image/png;base64,{base64.b64encode(open(keboola_gemini, "rb").read()).decode()}" style="width: 60px; margin-top: 30px;"></div>'
st.markdown(f'{gemini_html}', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: left;">
    <h4>Reply to a review with Gemini</h4>
</div>
""", unsafe_allow_html=True)

if selected_data['is_widget'].sum() == 1:
    gemini_data = selected_data[selected_data['is_widget'] == True]['text_in_english']
    review_text = gemini_data.iloc[0] if not gemini_data.empty else st.warning('No review found.')
    st.write(f'_Review:_\n\n{review_text}')

    selected_row = selected_data[selected_data['is_widget'] == True]

    if st.button('Generate response'):
        with st.spinner('🤖 Generating response, please wait...'):
            prompt = f"""
            You are given a review on London Eye, UK. Pretend you're a social media manager for the London Eye and write a short (3-5 sentence) response to this review. Only return the response.
            
            Review:
            {review_text}
            """
            response = generate(prompt)
            st.write(f"_Response:_\n\n{response}")
else:
    st.info('Select the review you want to respond to in the table above.')