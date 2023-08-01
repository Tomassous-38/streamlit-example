import streamlit as st
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup

def fetch_results(api_key, keyword, engine="google", location="Paris, Ile-de-France, France"):
    params = {
        "api_key": api_key,
        "q": keyword,
        "hl": "fr",  # Language French
        "gl": "fr",  # Country France
        "google_domain": "google.fr",
        "location": location,
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    urls = [item['link'] for item in results['organic_results'][:5]]

    contents = []
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.get_text()
            contents.append(content)
        except:
            contents.append("Failed to fetch content.")

    return urls, contents

st.title("Google Top 5 URLs Scraper")
st.write("Please enter your SERPapi key and the keyword.")

api_key = st.text_input("SERPapi Key")
keyword = st.text_input("Keyword")

if api_key and keyword:
    urls, contents = fetch_results(api_key, keyword)
    st.write("Top 5 URLs:")
    for url, content in zip(urls, contents):
        st.write(url)
        st.text_area("Content:", value=content, height=200)
