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
# Generate unique combinations of place_name and street
location['place_street'] = location['place_name'] + " - " + location['street']
location_options = location['place_street'].unique().tolist()
location_options.insert(0, "All")
# Generate unique list of brands
brand_options = location['brand'].dropna().unique().tolist()
brand_options.insert(0, "All")

# Set the title of the app
st.title("Location Experience")

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
        brand_selection = brand_options[1:]  # Exclude "All" from the actual selection
    
    # Location filter
    st.subheader("Location")
    location_selection = st.multiselect("Select Locations", options=location_options, default=["All"])
    if "All" in location_selection:
        location_selection = location_selection[1:]  # Exclude "All" from the actual selection


# Main layout
st.divider()
st.header("Overview")
col1, col2, col3 = st.columns(3, gap='medium')
st.markdown("<br>", unsafe_allow_html=True)

with col1:
    st.subheader("Sentiment")
    st.text("Bar chart from 0-5 (count)")

with col2:
    st.subheader("Location Map")
    st.text("Map visualization with location average sentiment")

with col3:
    st.subheader("Review Calendar")
    st.text("Calendar heatmap showing the number of reviews")

st.divider()
st.header("Entities")
col4, col5, col6, col7 = st.columns(4, gap='medium')
st.markdown("<br>", unsafe_allow_html=True)

with col4:
    st.subheader("Classification")
    st.text("Filters: sentence_category, sentence_category_group, sentence_topic")

with col5:
    st.subheader("Positive")
    st.text("Top 10 positive entities based on frequency")

with col6:
    st.subheader("Negative")
    st.text("Top 10 negative entities based on frequency")

with col7:
    st.subheader("Entities")
    st.text("Word cloud from entities")

st.divider()
st.header("Details")
st.text("Table with data from review_sentence and additional columns from location_review")

# Placeholder for the main content
st.write("""
    - Review Date filter (date range filter from-to)
    - Sentiment Score filter (value range filter from 0 to 5)
    - Brand filter (multi-selector)
    - Location filter (filtering location table based on "place_name - street", selection shall allow multiple)
    - Sentiment Score distribution (bar chart from 0-5 (count))
    - Map with location average sentiment (calculated from review)
    - Calendar heatmap with number of reviews shown on a calendar
    - Filters: (sentence_category, sentence_category_group, sentence_topic)
    - Top 10 positive entities (from review_entity) based on the frequency - where sentence_sentiment is Positive
    - Top 10 negative entities (from review_entity) based on the frequency - where sentence_sentiment is Negative
    - Word cloud from entities
    - Table with data from review_sentence and additional columns from location_review
""")
