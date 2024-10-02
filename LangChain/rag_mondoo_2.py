# Import dependencies
import json
import os
from dotenv import load_dotenv
from supabase import create_client, Client

from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_community.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Create LLM using Ollama for llama2:7b-chat
llm = Ollama(model="llama2:7b-chat")

# Helper function to generate prompt based on Mondoo data and user question
def generate_prompt(data):
    assets_section = ""
    for asset in data['assets']:
        # Ensure 'labels' is a dictionary (or an empty dictionary if None)
        labels = asset.get('labels', {}) or {}
        labels_section = ", ".join([f"{key}: {value}" for key, value in labels.items()])
        assets_section += f"""Asset MRN: {asset['mrn']}
        Name: {asset['name']}
        Platform: {asset['platform_name']}
        Error: {asset['error']}
        Score Updated At: {asset['score_updated_at']}
        Last Updated At: {asset['updated_at']}
        Labels: {labels_section}
        Annotations: {asset.get('annotations', '')}\n\n"""
    
    migrations_section = ""
    for migration in data.get('mondoo_migrations', []):
        migrations_section += f"""Migration ID: {migration['id']}
        Name: {migration['name']}
        Group ID: {migration['group_id']}
        Migrated At: {migration['migrated_at']}\n\n"""
    
    query_results_section = ""
    for query_item in data.get('query_result_items', []):
        query_results_section += f"""Asset MRN: {query_item['asset_mrn']}
        Query MRN: {query_item['query_mrn']}
        Title: {query_item['title']}
        MQL: {query_item['mql']}
        Data: {query_item['data']}\n\n"""
    
    prompt = f"""
    Mondoo Assets Data:
    {assets_section}
    
    Mondoo Migrations:
    {migrations_section}
    
    Query Result Items:
    {query_results_section}
    """
    
    # Write to file if a file path is provided
    with open('mondoo_prompt_output.txt', 'w') as f:
        f.write(prompt)

    return prompt


# Load Mondoo data
def load_mondoo():
    load_dotenv()
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
        json.dump(data, f, indent=4)

    st.success('Mondoo data loaded successfully!')
    return data

# Load Mondoo data initially
results = load_mondoo()

# Cache extracted data for retrieval
@st.cache_resource
def extract_information(data):
    documents = []
    for table in data:
        for item in data[table]:
            page_content = str(item)
            documents.append(Document(page_content=page_content, metadata={"table": table}))

    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    index = VectorstoreIndexCreator(
        # embedding=embeddings,
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

# Load Mondoo data
with st.spinner('Loading content...'):
    st.session_state.results = results

# User input
prompt_text = st.text_input("Enter your question here")

if prompt_text:

    # Display user message
    st.chat_message('user').markdown(prompt_text)
    st.session_state.messages.append({'role': 'user', 'content': prompt_text})

    if 'index' not in st.session_state:
        st.session_state.index = extract_information(st.session_state.results)

   # Generate the context based on Mondoo data
    context_prompt = generate_prompt(st.session_state.results)

    # Create a combined prompt, mixing the context with the user query
    combined_prompt = f"""
    Context:
    {context_prompt}

    User Query: {prompt_text}

    Response:
    """

    # Define a custom PromptTemplate (if needed for further customization)
    prompt_template = PromptTemplate(
        template="{combined_prompt}",
        input_variables=["combined_prompt"]
    )

    # Create the QA chain using LLM with custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=st.session_state.index.vectorstore.as_retriever()
    )

    # Run the QA chain using the combined prompt
    response = qa_chain.run({"query": combined_prompt})

    # Display the assistant response
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})
