import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
import os
import pdfplumber
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

st.set_page_config(page_title="Ford Chat bot", page_icon="🚗")
st.title("Ford Chat bot")
# Sidebar for chat history
st.sidebar.title("Chat History")
chat_history = st.sidebar.empty()

# Function to update chat history
def update_chat_history(user_input, response):
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({"user": user_input, "bot": response})
    chat_history.text_area("Chat History", value="\n".join(
        [f"User: {entry['user']}\nBot: {entry['bot']}" for entry in st.session_state["history"]]), height=300)

    # Sidebar for chat history
    st.sidebar.title("Chat History")
    chat_history = st.sidebar.empty()

    # Function to update chat history
    def update_chat_history(user_input, response):
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append({"user": user_input, "bot": response})
        chat_history.text_area("Chat History", value="\n".join(
            [f"User: {entry['user']}\nBot: {entry['bot']}" for entry in st.session_state["history"]]), height=300)

user_input = st.text_input("Ask your question:", key="user_input")

if user_input:
    # Assuming you have a function to get a response from the bot
    response = get_bot_response(user_input)
    update_chat_history(user_input, response)

    # Display the response
    st.write(response)

    # Auto-scroll to the bottom of the page
    st.experimental_rerun()


# Load FAISS index
vectorstore = FAISS.load_local("f150_vectorstore", OpenAIEmbeddings())

# Query function using Groq
def answer_query(query):
    retriever = vectorstore.as_retriever()
    llm = ChatGroq()  # Initialize Groq model
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    response = qa_chain.run(query)
    return response


st.set_page_config(page_title="PDF Chatbot", page_icon="📄")

st.title("PDF Chatbot")
st.markdown("Ask any question about the content of the PDFs.")

# Query input
user_query = st.text_input("Ask your question:")

if user_query:
    with st.spinner("Searching..."):
        try:
            response = answer_query(user_query)
            st.success("Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")