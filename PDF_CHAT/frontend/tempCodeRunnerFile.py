import streamlit as st
import requests

st.set_page_config(page_title="Doc Chatbot")
st.title("Document Q&A Chatbot")

query = st.text_input("Ask a question based on the document:")
if st.button("Get Answer") and query.strip():
    with st.spinner("Thinking..."):
        try:
            response = requests.post("http://localhost:5000/query", json={"query": query})
            if response.status_code == 200:
                st.success(response.json()["answer"])
            else:
                st.error("Error from server.")
        except Exception as e:
            st.error(f"Connection failed: {e}")
