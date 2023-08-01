import streamlit as st
from serpapi import GoogleSearch

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
    return urls

st.title("Google Top 5 URLs Scraper")
st.write("Please enter your SERPapi key and the keyword.")

api_key = st.text_input("SERPapi Key")
keyword = st.text_input("Keyword")

if api_key and keyword:
    urls = fetch_results(api_key, keyword)
    st.write("Top 5 URLs:")
    for url in urls:
        st.write(url)
