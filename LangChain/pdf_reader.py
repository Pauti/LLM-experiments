import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import tempfile

# Load API key
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Load and process PDF
@st.cache_resource
def load_and_process_pdf(pdf_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings)

    # Remove the temporary file
    os.unlink(tmp_file_path)

    return vectorstore

# Conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Handle user input
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write("Human: " + message.content)
        else:
            st.write("AI: " + message.content)

# Handle application
def main():
    st.set_page_config(page_title="Chat with your PDF", page_icon=":books:")
    st.header("Chat with your PDF:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            vectorstore = load_and_process_pdf(uploaded_file)
            st.session_state.conversation = get_conversation_chain(vectorstore)
        st.success("PDF processed successfully!")

        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            handle_user_input(user_question)

if __name__ == '__main__':
    main()