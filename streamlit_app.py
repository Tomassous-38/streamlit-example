import streamlit as st
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_results(api_key, keyword, debug=False, location="Paris, Paris, Ile-de-France, France"):
    if debug: st.write("Fetching results...")
    params = {
        "api_key": api_key,
        "engine": "google",
        "q": keyword,
        "location": location,
        "google_domain": "google.fr",
        "gl": "fr",
        "hl": "fr"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    time.sleep(4)
    urls = [item['link'] for item in results['organic_results'][:5]]
    if debug: st.write("Fetched URLs:", urls)
    return urls

def get_text_from_url(url, debug=False):
    if debug: st.write(f"Getting text from URL: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

def summarize_url(url, openai_api_key, debug=False):
    if debug: st.write(f"Processing URL: {url}")
    text = get_text_from_url(url, debug=debug)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.8)
    chain_summarize = load_summarize_chain(llm, chain_type="map_reduce")
    documents = text_splitter.create_documents([text])
    summaries = []
    for doc in documents:
        if isinstance(doc, tuple):
            doc = {'page_content': doc[0]}
        summarized_texts = chain_summarize.run([doc])
        summaries.append(summarized_texts)
    return url, summaries

def summarize_text(urls, openai_api_key, debug=False):
    with ThreadPoolExecutor() as executor:
        future_summaries = {executor.submit(summarize_url, url, openai_api_key, debug): url for url in urls}
        summaries = [future.result() for future in as_completed(future_summaries)]
    if debug: st.write("Summaries created.")
    return summaries

def main():
    st.title("Google Top 5 URLs Scraper & Summarizer")
    api_key = st.text_input("SERPapi Key")
    keyword = st.text_input("Keyword")
    openai_api_key = st.text_input("OpenAI API Key")
    debug_mode = st.checkbox("Debug Mode")

    if st.button("Send") and api_key and keyword and openai_api_key:
        urls = fetch_results(api_key, keyword, debug=debug_mode)
        st.write("Top 5 URLs:")
        summaries = summarize_text(urls, openai_api_key, debug=debug_mode)
        for url, summary in summaries:
            st.write(f"URL: {url}")
            st.write(f"Summary: {summary}")

if __name__ == "__main__":
    main()
