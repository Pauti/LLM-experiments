# Import dependencies
import json
import os
from dotenv import load_dotenv
from supabase import create_client, Client

from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Create LLM using ChatOpenAI for GPT-4
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

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
          page_content = str(item)
          documents.append(Document(page_content=page_content, metadata={"table": table}))

  embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

  index = VectorstoreIndexCreator(
      embedding=embeddings,
      text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
  ).from_documents(documents)

  return index

# Create the custom prompt template
prompt_template = """
You are analyzing a JSON output from Mondoo, a security and compliance platform. The JSON contains information about various assets and their properties.
Your task is to interpret the data and provide insights based on the following structure:

1. **Asset Details:**
   - Name: [Asset Name]
   - Platform: [Platform Name]
   - Unique Identifier (MRN): [MRN]
   - Last Updated: [Updated At]
   - Error Messages: [Error]

2. **Metadata:**
   - Labels: [Key-Value Pairs]
   - Annotations: [Annotations]

3. **Query Results:**
   - Query Title: [Title]
   - Query Data: [Data]
   - Additional Insights: [Interpretation of Data]

Use this structure to provide a comprehensive analysis of each asset and its associated data.

Context:
{context}

Question:
{question}

Answer in a clear and concise manner:
"""

prompt = PromptTemplate(
  template=prompt_template,
  input_variables=["context", "question"]
)

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
  # Store user message in session state
  st.session_state.messages.append({'role': 'user', 'content': prompt_text})

  # Check if index is already in session state to avoid reprocessing
  if 'index' not in st.session_state:
      st.session_state.index = extract_information(st.session_state.results)

  # Create the QA chain using ChatOpenAI and custom prompt
  qa_chain = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type='stuff',
      retriever=st.session_state.index.vectorstore.as_retriever(),
      chain_type_kwargs={
          "prompt": prompt,
          "document_variable_name": "context"
      }
  )

  # Get response from LLM
  response = qa_chain.run(prompt_text)
  # Show assistant response
  st.chat_message('assistant').markdown(response)
  # Store assistant response in session state
  st.session_state.messages.append({'role': 'assistant', 'content': response})