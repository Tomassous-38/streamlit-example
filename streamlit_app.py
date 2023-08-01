import streamlit as st
import requests
from langchain.text_splitter import TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate, LLMChain, OpenAI
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
    urls = [item['link'] for item in results['organic_results'][:5]]
    return urls

def get_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

def summarize_text(urls, openai_api_key):
    text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=200)
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.8)
    chain_summarize = load_summarize_chain(llm, chain_type="map_reduce")

    summaries = []
    for url in urls:
        text = get_text_from_url(url)
        splitted_texts = text_splitter.split_text(text)
        summarized_text = chain_summarize.run(splitted_texts)
        summaries.append((url, summarized_text))

    return summaries

def main():
    st.title("Google Top 5 URLs Scraper & Summarizer")
    api_key = st.text_input("SERPapi Key")
    keyword = st.text_input("Keyword")
    openai_api_key = st.text_input("OpenAI API Key")

    if api_key and keyword and openai_api_key:
        urls = fetch_results(api_key, keyword)
        st.write("Top 5 URLs:")
        summaries = summarize_text(urls, openai_api_key)
        for url, summary in summaries:
            st.write(f"URL: {url}")
            st.write(f"Summary: {summary}")

if __name__ == "__main__":
    main()
