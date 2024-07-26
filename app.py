import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os 
import jwt
import vertexai
import base64
from wordcloud import WordCloud
from PIL import Image
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel
from keboola_streamlit import KeboolaStreamlit

PROJECT = 'keboola-ai'
LOCATION = 'us-central1'
MODEL_NAME = 'gemini-1.5-pro-preview-0409'
IMAGE_PATH = os.path.dirname(os.path.abspath(__file__))
KEBOOLA_LOGO_PATH = IMAGE_PATH + "/static/keboola_logo.png"
KEBOOLA_GEMINI_PATH = IMAGE_PATH + "/static/keboola_gemini.png"

CREDENTIALS = service_account.Credentials.from_service_account_info(
    jwt.decode(st.secrets['ENCODED_TOKEN'], 'keboola', algorithms=['HS256'])
)

st.set_page_config(layout='wide')

@st.cache_data
def read_data(table_name):
    df = pd.read_csv(table_name)
    return df

def color_for_value(value):
    if value < 3:
        return '#EA4335'
    elif 3 <= value <= 4.2:
        return '#FBBC05' 
    else:
        return '#34A853' 

def sentiment_color(sentiment):
    if sentiment == "Positive":
        return "color: #34A853"
    if sentiment == "Mixed":
        return "color: #FBBC05"
    else:
        return "color: #EA4335"

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
        st.error(f"An error occurred during content generation. Please try again.")
        return ''

logo_html = f'<div style="display: flex; justify-content: flex-end;"><img src="data:image/png;base64,{base64.b64encode(open(KEBOOLA_LOGO_PATH, "rb").read()).decode()}" style="width: 200px; margin-bottom: 10px;"></div>'
st.markdown(f"{logo_html}", unsafe_allow_html=True)

location = read_data('data/in/tables/out.c_review_model.location.csv')
location_review = read_data('data/in/tables/in.c-qsr_model.location_review.csv')
review_sentence = read_data('data/in/tables/in.c-qsr_model.review_sentence.csv')
review_entity = read_data('data/in/tables/in.c-qsr_model.review_entity.csv')

# Clean up datetimes
location_review['review_date'] = pd.to_datetime(location_review['review_date'], format='mixed').dt.tz_localize(None)

# Generate unique list of brands
brand_options = location['brand'].dropna().unique().tolist()
brand_options.insert(0, "All")

# Set the title of the app
st.title("Location Experience")

# SIDEBAR

# Top row for filters
with st.sidebar:
    st.header("Filters")
    
    # Review Date filter
    st.subheader("Review Date")
    review_date = st.date_input("Select Date Range", [])
    
    # Sentiment Score filter
    st.subheader("Sentiment Score")
    sentiment_score = st.slider("Select Sentiment Score Range", 0.0, 5.0, (0.0, 5.0))
    
    # Brand filter
    st.subheader("Brand")
    brand_selection = st.multiselect("Select Brands", options=brand_options, default=["All"])
    if "All" in brand_selection:
        brand_selection = brand_selection[1:]
    
    # Filter locations based on selected brand
    if "All" in brand_selection:
        filtered_locations = location
    else:
        filtered_locations = location[location['brand'].isin(brand_selection)]
    
    # Generate unique combinations of place_name and street
    filtered_locations['place_street'] = filtered_locations['place_name'] + " - " + filtered_locations['street']
    location_options = filtered_locations['place_street'].unique().tolist()
    location_options.insert(0, "All")
    
    # Location filter
    st.subheader("Location")
    location_selection = st.multiselect("Select Locations", options=location_options, default=["All"])
    if "All" in location_selection:
        location_selection = location_options[1:]  # Exclude "All" from the actual selection

# DATA FILTERING

# Filter location_review dataset based on the selected filters
filtered_reviews = location_review.copy()

# Apply Review Date filter
if review_date:
    start_date, end_date = review_date
    filtered_reviews = filtered_reviews[(filtered_reviews['review_date'] >= pd.to_datetime(start_date)) & (filtered_reviews['review_date'] <= pd.to_datetime(end_date))]

# Apply Sentiment Score filter
filtered_reviews = filtered_reviews[(filtered_reviews['rating'] >= sentiment_score[0]) & (filtered_reviews['rating'] <= sentiment_score[1])]

# Apply Brand and Location filters
if brand_selection and location_selection:
    if "All" not in brand_selection:
        place_ids = filtered_locations[filtered_locations['place_street'].isin(location_selection)]['place_id'].unique()
        filtered_reviews = filtered_reviews[filtered_reviews['place_id'].isin(place_ids)]

# Apply filters to review_entity dataset
filtered_entities = review_entity[review_entity['review_id'].isin(filtered_reviews['review_id'])]

# Main layout
st.divider()
st.header("Overview")
col1, col2 = st.columns([1, 2], gap='medium')
st.markdown("<br>", unsafe_allow_html=True)

with col1:
    st.subheader("Sentiment")
    if not filtered_reviews.empty:
        sentiment_distribution = filtered_reviews['rating'].value_counts().sort_index()
        fig = px.bar(sentiment_distribution, x=sentiment_distribution.index, y=sentiment_distribution.values, labels={'x':'Rating', 'y':'Count'})
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected filters.")

with col2:
    st.subheader("Location Map")
    if not filtered_reviews.empty:
        # Merge location and filtered reviews to get average sentiment per location
        merged_data = filtered_reviews.merge(location, on='place_id')
        avg_sentiment = merged_data.groupby(['formatted_address', 'latitude', 'longitude'])['rating'].mean().reset_index()
        avg_sentiment['text'] = avg_sentiment['formatted_address'] + ': ' + avg_sentiment['rating'].round(2).astype(str)
        
        fig = px.scatter_mapbox(
            avg_sentiment, 
            lat='latitude', 
            lon='longitude', 
            text='text', 
            size='rating', 
            color='rating', 
            color_continuous_scale=px.colors.cyclical.IceFire, 
            size_max=15
        )

        # Center and zoom map on selected locations
        if not avg_sentiment.empty:
            lat_center = avg_sentiment['latitude'].mean()
            lon_center = avg_sentiment['longitude'].mean()
            fig.update_layout(
                mapbox_style="carto-positron",
                mapbox_center={"lat": lat_center, "lon": lon_center},
                mapbox_zoom=10
            )
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected filters.")

st.divider()
st.header("Entities")
col4, col5, col6, col7 = st.columns(4, gap='medium')
st.markdown("<br>", unsafe_allow_html=True)

with col4:
    st.subheader("Classification")
    sentence_category = st.multiselect("Select Sentence Categories", options=review_sentence['sentence_category'].dropna().unique().tolist())

    # Filter category groups and topics based on selected categories
    filtered_review_sentence = review_sentence.copy()
    if sentence_category:
        filtered_review_sentence = filtered_review_sentence[filtered_review_sentence['sentence_category'].isin(sentence_category)]

    sentence_category_group = st.multiselect("Select Sentence Category Groups", options=filtered_review_sentence['sentence_category_group'].dropna().unique().tolist())
    
    # Further filter topics based on selected categories and category groups
    if sentence_category_group:
        filtered_review_sentence = filtered_review_sentence[filtered_review_sentence['sentence_category_group'].isin(sentence_category_group)]
        
    sentence_topic = st.multiselect("Select Sentence Topics", options=filtered_review_sentence['sentence_topic'].dropna().unique().tolist())

# Apply filters to filtered_entities based on selected categories, category groups, and topics
if sentence_category:
    filtered_entities = filtered_entities[filtered_entities['sentence_category'].isin(sentence_category)]
if sentence_category_group:
    filtered_entities = filtered_entities[filtered_entities['sentence_category_group'].isin(sentence_category_group)]
if sentence_topic:
    filtered_entities = filtered_entities[filtered_entities['sentence_topic'].isin(sentence_topic)]

# Filter detailed_data based on the same filters
detailed_data = review_sentence.merge(location_review, on='review_id', suffixes=('_sentence', '_review'))
if sentence_category:
    detailed_data = detailed_data[detailed_data['sentence_category'].isin(sentence_category)]
if sentence_category_group:
    detailed_data = detailed_data[detailed_data['sentence_category_group'].isin(sentence_category_group)]
if sentence_topic:
    detailed_data = detailed_data[detailed_data['sentence_topic'].isin(sentence_topic)]



# Add a 'select' column for the selector widget
detailed_data['select'] = ""

with col5:
    st.subheader("Positive Entities")
    if not filtered_entities.empty:
        positive_entities = filtered_entities[filtered_entities['sentence_sentiment'] == 'Positive']['entity'].value_counts().head(10).sort_values(ascending=True)
        if not positive_entities.empty:
            fig_positive = px.bar(positive_entities, x=positive_entities.values, y=positive_entities.index, orientation='h')
            fig_positive.update_layout(xaxis_title=None, yaxis_title=None)
            fig_positive.update_traces(marker_color='#34A853')  # Green color
            st.plotly_chart(fig_positive)
        else:
            st.write("No positive entities available for the selected filters.")
    else:
        st.write("No data available for the selected filters.")

with col6:
    st.subheader("Negative Entities")
    if not filtered_entities.empty:
        negative_entities = filtered_entities[filtered_entities['sentence_sentiment'] == 'Negative']['entity'].value_counts().head(10).sort_values(ascending=True)
        if not negative_entities.empty:
            fig_negative = px.bar(negative_entities, x=negative_entities.values, y=negative_entities.index, orientation='h')
            fig_negative.update_layout(xaxis_title=None, yaxis_title=None)
            fig_negative.update_traces(marker_color='#FF5733')  # Reddish-orange color
            st.plotly_chart(fig_negative)
        else:
            st.write("No negative entities available for the selected filters.")
    else:
        st.write("No data available for the selected filters.")

with col7:
    st.subheader("Word Cloud")
    if not filtered_entities.empty:
        wordcloud = WordCloud(width=400, height=800, background_color='white').generate(" ".join(filtered_entities['entity']))
        plt.figure(figsize=(10, 20))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.write("No data available for the selected filters.")

st.divider()
st.header("Review Feedback")

if not detailed_data.empty:
    selected_data = st.data_editor(detailed_data.style.map(
            sentiment_color, subset=["sentence_sentiment"]
        ),
                column_config={'select': 'Select',
                               'sentence_sentiment': 'Sentiment',
                               'sentence_text': 'Sentence',
                               'sentence_category': 'Category',
                               'sentence_category_group': 'Category Group',
                               'sentence_topic': 'Topic',
                               'entities': 'Entities',
                               'place_name': 'Location',
                               'author': 'Author',
                               'rating': 'Rating',
                               'review_date': 'Date',
                               'sentiment': 'Overall Sentiment'
                                }, height=500,
                 disabled=['review_id',
                           'place_id'],
                use_container_width=True, hide_index=True)
    #st.data_editor(detailed_data[display_columns], disabled=False, hide_index=True, use_container_width=True)
else:
    st.write("No data available for the selected filters.")


