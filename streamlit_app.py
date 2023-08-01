import streamlit as st 
import requests
from bs4 import BeautifulSoup
import re

from langchain.text_splitter import TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate, LLMChain, OpenAI

def get_latest_results(query, api_key):
    params = {
        "q": query,
        "location": "United States",
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "tbs": "qdr:d",
        "api_key": api_key,
    }

    response = requests.get("https://serpapi.com/search", params)
    results = response.json()

    urls = [r["link"] for r in results["organic_results"]][:5] # Limit to first 5 results

    return urls

def scrape_and_chunk_urls(urls):
    text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = []

    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = ' '.join([re.sub(r'\s+', ' ', p.get_text()) for p in soup.find_all('p')])

            chunks += text_splitter.split_text(text)
        except:
            print(f"Failed to download and parse article: {url}")

    return chunks

def summarize_text(to_summarize_texts, openai_api_key):
    summarized_texts = []

    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.8)
    chain_summarize = load_summarize_chain(llm, chain_type="map_reduce")
    
    for text in to_summarize_texts:
        # Summarize chunks here
        summarized_text = chain_summarize.run(text)

        summarized_texts.append(summarized_text)

    return summarized_texts

def main():
    st.title('Content Summarizer')
    st.markdown("## Please input your API keys")

    serpapi_key = st.text_input("Insert your SerpAPI key here: ", type="password")
    openai_api_key = st.text_input("Insert your OpenAI api key: ", type="password")
    user_query = st.text_input("Make me a summary about: ")

    if st.button('Submit'):
        urls = get_latest_results(user_query, serpapi_key)
        chunks = scrape_and_chunk_urls(urls)
        summarized_texts = summarize_text(chunks, openai_api_key)
        
        for summary in summarized_texts:
            st.write(f"❇️ {summary}")
            st.markdown("\n\n")

if __name__ == "__main__":
    main()
