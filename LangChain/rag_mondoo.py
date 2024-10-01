# Import langchain dependencies
import json
import os
import time
import threading
from dotenv import load_dotenv
from supabase import create_client, Client

from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA, LLMChain
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from wxai_langchain.llm import LangChainInterface
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# Create LLM
llm = Ollama(model="llama2:7b-chat")

def load_mondoo():
    load_dotenv()
    # Supabase PostgreSQL connection
    base_url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    supabase: Client = create_client(base_url, key)

    tables = [
        'assets',
        'check_result_items',
        'mondoo_migration_locks',
        'mondoo_migrations',
        'query_result_items'
    ]

    data = {}
    for table in tables:
        response = supabase.table(table).select("*").execute()
        data[table] = response.data

    with open('mondoo_data.json', 'w') as f:
        json.dump(data, f)

    st.success('Mondoo data loaded successfully!')

    return data
# Load Mondoo data initially
results = load_mondoo()

@st.cache_resource
def extract_information(data):

    documents = []
    for table in data:
        for item in data[table]:
            # Ensure page_content is always a string
            page_content = str(item)  # Convert item to string to avoid validation errors
            documents.append(Document(page_content=page_content, metadata={"table": table}))

    index = VectorstoreIndexCreator(
        embedding=OllamaEmbeddings(),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
    ).from_documents(documents)
    return index

# Streamlit UI
st.title("Talk to your Mondoo!")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Load Mondoo data (it will be refreshed in the background every minute)
with st.spinner('Loading content...'):
    st.session_state.results = results

# User input
prompt = st.text_input("Enter your question here")

if prompt:
    # Display user message
    st.chat_message('user').markdown(prompt)
    # Store user message in session state
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    st.session_state.index = extract_information(st.session_state.results)

    # Create the QA chain using the index from session state
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=st.session_state.index.vectorstore.as_retriever(),
        input_key='question',
    )

    # Get response from LLM
    response = qa_chain.run({"question": prompt})
    # Show assistant response
    st.chat_message('assistant').markdown(response)
    # Store assistant response in session state
    st.session_state.messages.append({'role': 'assistant', 'content': response})
