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
    """Fetch Google search results for a given keyword"""
    if debug:
        st.write("Fetching results...")
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
    time.sleep(4)  # Consider async or better timing strategies if possible
    urls = [item['link'] for item in results['organic_results'][:5]]
    if debug:
        st.write("Fetched URLs:", urls)
    return urls

def get_text_from_url(url, debug=False):
    """Retrieve the full text content from a URL"""
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for failed requests
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

def summarize_text(urls, openai_api_key, debug=False):
    """Summarize the text from a list of URLs using LangChain"""
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    chain = load_summarize_chain(llm, chain_type="stuff")
    summaries = []
    for url in urls:
        text = get_text_from_url(url, debug=debug)
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
        summary = chain.run(docs)
        summaries.append((url, summary))
    return summaries

def custom_summary(summaries, keyword):
    """Combine summaries into a custom summary"""
    full_text = ' '.join([summary for _, summary in summaries])
    custom_prompt = f"En gardant toutes les informations sur {keyword} résume ce texte: {full_text}"
    prompt_template = PromptTemplate(template=custom_prompt)
    llm = OpenAI(engine="gpt-4", max_tokens=2000)
    response = llm.complete(prompt_template)
    return response

def main():
    """Main function to run the Streamlit app"""
    st.title("Google Top 5 URLs Scraper & Summarizer")
    api_key = st.text_input("SERPapi Key")
    keyword = st.text_input("Keyword")
    openai_api_key = st.text_input("OpenAI API Key", type="password")  # Hide API key input
    debug_mode = st.checkbox("Debug Mode")

    if st.button("Send") and api_key and keyword and openai_api_key:
        try:
            urls = fetch_results(api_key, keyword, debug=debug_mode)
            st.write("Top 5 URLs:")
            summaries = summarize_text(urls, openai_api_key, debug=debug_mode)
            final_summary = custom_summary(summaries, keyword)
            st.write(f"Final Summary: {final_summary}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")  # Proper error handling

if __name__ == "__main__":
    main()
