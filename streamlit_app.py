import streamlit as st
from bs4 import BeautifulSoup
from google-search-results import GoogleSearch
from langchain.text_splitter import TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate, LLMChain, OpenAI
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
    print(results) # Prints the entire JSON response
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

    if st.button("Send") and api_key and keyword and openai_api_key:
        urls = fetch_results(api_key, keyword)
        st.write("Top 5 URLs:")
        summaries = summarize_text(urls, openai_api_key)
        for url, summary in summaries:
            st.write(f"URL: {url}")
            st.write(f"Summary: {summary}")

if __name__ == "__main__":
    main()
