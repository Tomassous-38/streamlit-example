import streamlit as st
import requests
import json
import numpy as np

from bs4 import BeautifulSoup
from langchain.text_splitter import TokenTextSplitter
from langchain import PromptTemplate, LLMChain, OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# The Document class is defined with the page_content attribute
class Document:
    def __init__(self, title, text):
        self.title = title
        self.page_content = text
        self.metadata = {"stop": []}

def get_latest_results(query, api_key):
    # Scrapes google search results
    params = {
        "q": query,
        "location": "United States",
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "api_key": api_key,
    }

    response = requests.get("https://serpapi.com/search", params)
    results = json.loads(response.text)
    urls = [r["link"] for r in results["organic_results"]][:5]  # limit to first 5 results
    return urls

def scrape_urls(urls):
    # Initialize the text_splitter before using it
    text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=200)
    parsed_texts = []

    # iterate over each URL
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            article_text = soup.get_text()

            # split text into chunks of 3k tokens
            splitted_texts = text_splitter.split_text(article_text)
            if not splitted_texts:
                print(article_text)

            # Append tuple of splitted text and URL to the list
            parsed_texts.append((splitted_texts, url))

        except Exception as e:
            print(f"Failed to download and parse article: {url}, Error: {e}")

    return parsed_texts

def summarize_text(to_summarize_texts, openai_api_key):
    summarized_texts_titles_urls = []
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.8)
    chain_summarize = load_summarize_chain(llm, chain_type="map_reduce")

    # Define prompt that generates titles for summarized text
    prompt = PromptTemplate(
            input_variables=["text"],
            template="Write an appropriate, clickbaity news article title in less than 70 characters for this text: {text}"
    )

    for to_summarize_text, url in to_summarize_texts:
        # Convert each text string to a Document object
        to_summarize_text = [Document('Dummy Title', text) for text in to_summarize_text]
        if not to_summarize_text:
            print(f"No text to summarize for URL: {url}")
            continue

        # Summarize chunks here
        summarized_text = chain_summarize.run(to_summarize_text)
        chain_prompt = LLMChain(llm=llm, prompt=prompt)
        clickbait_title = chain_prompt.run(summarized_text)

        summarized_texts_titles_urls.append((clickbait_title, summarized_text, url))

    return summarized_texts_titles_urls

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
