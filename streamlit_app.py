import streamlit as st
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
from langchain.docstore.document import Document
import time

def fetch_results(api_key, keyword, location="Paris, Paris, Ile-de-France, France"):
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
    # Wait for 4 seconds to ensure data processing
    time.sleep(4)
    urls = [item['link'] for item in results['organic_results'][:5]]
    return urls

def get_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

def summarize_text(urls, openai_api_key):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.8)
    chain_summarize = load_summarize_chain(llm, chain_type="map_reduce")

    summaries = []
    for url in urls:
        text = get_text_from_url(url)
        documents = text_splitter.create_documents([text])
        summarized_texts = []
        for doc in documents:
            # Ensure that 'doc' is in the expected format
            if isinstance(doc, tuple):
                doc = {'page_content': doc[0]}
            summary_chunk = chain_summarize.run([doc])[0]
            summarized_texts.append(summary_chunk)

        # Combine the summarized chunks into a single summary
        final_summary = " ".join(summarized_texts)
        summaries.append((url, final_summary))

    return summaries



def main():
    st.title("Google Top 5 URLs Scraper & Summarizer")
    api_key = st.text_input("SERPapi Key")
    keyword = st.text_input("Keyword")
    openai_api_key = st.text_input("OpenAI API Key")

    if st.button("Send") and api_key and keyword and openai_api_key:
        urls = fetch_results(api_key, keyword)
        st.write("Top 5 URLs:")
        summaries = summarize_text(urls, openai_api_key)
        for url, summary in summaries:
            st.write(f"URL: {url}")
            for summarized_text in summary:
                st.write(f"Summary: {summarized_text}")

if __name__ == "__main__":
    main()
