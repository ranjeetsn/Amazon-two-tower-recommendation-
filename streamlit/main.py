import streamlit as st
import requests
import argparse

parser = argparse.ArgumentParser(description='Streamlit App')
parser.add_argument('--host', default='0.0.0.0', required=False, \
                    help='The host address of the FastAPI server')
args = parser.parse_args()

st.title('Amazon Product Recommendation System')


query = st.text_input("Enter your product query:", "")

if st.button('Search') and query:
    response = requests.post(
        f"http://{args.host}:8000/search",
        json={"query": query}, 
        headers={'Content-Type': 'application/json'}
    )
    print(f"http://{args.host}:8000/search")
    if response.status_code == 200:
        results = response.json().get('results', [])
        if results:
            for idx, result in enumerate(results):
                st.write(f"{idx + 1}. {result}")
        else:
            st.write("No results found.")
    else:
        st.write("Failed to fetch results:", response.status_code)
