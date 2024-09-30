# import langchain dependencies
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OllamaEmbeddings # Ollama embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# bring in streamlit for UI development
import streamlit as st
# bring in watsonx interface
from wxai_langchain.llm import LangChainInterface
# import ollama for language model
from langchain_community.llms import Ollama

# Create LLM
llm = Ollama(model="llama2:7b-chat")

# Function to laod PDF file
@st.cache_resource
def load_pdf(pdf_name):

    loaders = [PyPDFLoader(pdf_name)]
    # create index / vector store
    index = VectorstoreIndexCreator(
        embedding = OllamaEmbeddings(),
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
    ).from_loaders(loaders)

    return index

# Load PDF file
pdf_name = st.text_input("Enter the path to the PDF file:", value="/Users/paulstrebenitzer/Projects/AI/LangChain/Cats.pdf")

with st.spinner('Loading index...'):
  index = load_pdf(pdf_name)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = 'stuff',
    retriever = index.vectorstore.as_retriever(),
    input_key = 'question',
)

# setup app title
st.title("Ask me anything!")

# Setup session state message variable to hold all messages
if 'messages' not in st.session_state:
  st.session_state.messages = []

# Display all historical messages
for message in st.session_state.messages:
  st.chat_message(message['role']).markdown(message['content'])

# Build prompt input template to display prompts
prompt = st.chat_input("Enter your question here")

# If user hits enter, run the following code
if prompt:
  # diaply prompt message
  st.chat_message('user').markdown(prompt)
  # Store prompt message in session state
  st.session_state.messages.append({'role': 'user', 'content': prompt})
  # send prompt to llm
  response = qa_chain.run(prompt)
  # show llm response
  st.chat_message('assistant').markdown(response)
  st.session_state.messages.append({'role': 'assistant', 'content': response})
