# Import langchain dependencies
import os
from dotenv import load_dotenv
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA, LLMChain
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from wxai_langchain.llm import LangChainInterface
from langchain_community.llms import Ollama
import requests
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# Create LLM
llm = Ollama(model="llama2:7b-chat")

# Function to perform web search using Bing Search API
def web_search(query):
    load_dotenv()
    subscription_key = os.getenv('SUBSCRIPTION_KEY')
    search_url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    return search_results

# Function to extract relevant information from search results
@st.cache_resource
def extract_information(search_results):
    documents = []
    for result in search_results.get("webPages", {}).get("value", []):
        documents.append(Document(page_content=result["snippet"], metadata={"title": result["name"], "url": result["url"]}))
    
    index = VectorstoreIndexCreator(
        embedding=OllamaEmbeddings(),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
    ).from_documents(documents)
    return index

# Streamlit UI
st.title("LangChain Web Search App")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'index' not in st.session_state:
    st.session_state.index = None

# Display all historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# User input
prompt = st.chat_input("Enter your question here")

# Checkbox to decide whether to perform web search
update_knowledge = st.checkbox('Search the web to update information')

if prompt:
    # Display user message
    st.chat_message('user').markdown(prompt)
    # Store user message in session state
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Perform web search and update index if checkbox is selected
    if update_knowledge is True:
        with st.spinner('Loading content...'):
            search_results = web_search(prompt)
            st.session_state.index = extract_information(search_results)
        with st.spinner('Generating answer...'):
            # Create the QA chain using the index from session state
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=st.session_state.index.vectorstore.as_retriever(),
                input_key='question',
            )
    else:
        st.write("Using cached information to answer your question.")
        # Create a prompt template for LLMChain
        prompt_template = PromptTemplate(input_variables=["question"], template="{question}")
        qa_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Get response from LLM
    response = qa_chain.run({"question": prompt})
    # Show assistant response
    st.chat_message('assistant').markdown(response)
    # Store assistant response in session state
    st.session_state.messages.append({'role': 'assistant', 'content': response})