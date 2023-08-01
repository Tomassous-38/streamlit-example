from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import streamlit as st
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import time

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
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except Exception as e:
        if debug:
            print(f"An error occurred while scraping {url}: {e}")
        return None

def summarize_text(urls, openai_api_key, debug=False):
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")  # Using map_reduce chain
    summaries = []
    for url in urls:
        text = get_text_from_url(url, debug=debug)
        if text is None:
            continue  # If an error occurred, ignore this URL and continue with the next one
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
        summary = chain.run(docs)
        summaries.append((url, summary))
    return summaries

def custom_summary(summaries, keyword):
    full_text = ' '.join([summary for _, summary in summaries])
    custom_prompt = f"En gardant toutes les informations sur {keyword} résume ce texte: {full_text}"
    prompt_template = PromptTemplate(template=custom_prompt)
    llm = OpenAI(engine="gpt-4", max_tokens=2000)
    response = llm.complete(prompt_template)
    return response

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
        final_summary = custom_summary(summaries, keyword)
        st.write(f"Final Summary: {final_summary}")

if __name__ == "__main__":
    main()
