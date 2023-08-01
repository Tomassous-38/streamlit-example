import streamlit as st
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
import requests
import openai
import textwrap

st.title("Streamlit App for SERP Scraping and Content Summarization")

serp_api_key = st.text_input("Enter your SERPapi Key:", type="password")
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

query = st.text_input("Enter your search query:")

if st.button("Proceed") and query and serp_api_key and openai_api_key:
    # Search Google with SERPapi
    params = {
        "engine": "google",
        "q": query,
        "num": 5,
        "api_key": serp_api_key,
    }
    search = GoogleSearch(params)
    results = search.get_dict().get("organic_results", [])[:5]

    urls_content = []

    # Scrape the URLs using BeautifulSoup
    for result in results:
        url = result['link']
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        urls_content.append(text)

    # Chunk the URL Content into Tokens and Summarize
    openai.api_key = openai_api_key

    summaries = []

    for content in urls_content:
        chunks = textwrap.wrap(content, 3000) # Splitting the content into 3000 character chunks

        for chunk in chunks:
            # Customize the summarization model call according to your requirements
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt="Summarize the following text: " + chunk,
                max_tokens=150
            )
            summary = response.choices[0].text
            summaries.append(summary)

    # Display the summaries
    for summary in summaries:
        st.write(summary)
