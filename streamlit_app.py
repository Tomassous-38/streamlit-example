import streamlit as st
import requests
from langchain.text_splitter import TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
from bs4 import BeautifulSoup

def fetch_results(api_key, keyword, location="Paris, Ile-de-France, France"):
    params = {
        "api_key": api_key,
        "q": keyword,
        "hl": "fr",  # Language French
        "gl": "fr",  # Country France
        "google_domain": "google.fr",
        "location": location,
    }
    response = requests.get("https://serpapi.com/search", params)
    results = response.json()
    return [item['link'] for item in results.get('organic_results', [])[:5]]

def get_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

def summarize_text(urls, openai_api_key):
    text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=200)
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.8)
    chain_summarize = load_summarize_chain(llm, chain_type="map_reduce")

    return [(url, chain_summarize.run(text_splitter.split_text(get_text_from_url(url)))) for url in urls]

def main():
    st.title("Google Top 5 URLs Scraper & Summarizer")
    api_key = st.text_input("SERPapi Key")
    keyword = st.text_input("Keyword")
    openai_api_key = st.text_input("OpenAI API Key")

    if st.button('Send'):
        if all([api_key, keyword, openai_api_key]):
            urls = fetch_results(api_key, keyword)
            if urls:
                st.write("Top 5 URLs:")
                summaries = summarize_text(urls, openai_api_key)
                for url, summary in summaries:
                    st.write(f"URL: {url}")
                    st.write(f"Summary: {summary}")
            else:
                st.write("No results found. Please check the parameters.")
                if st.button('Retry'):
                    st.experimental_rerun()

if __name__ == "__main__":
    main()
