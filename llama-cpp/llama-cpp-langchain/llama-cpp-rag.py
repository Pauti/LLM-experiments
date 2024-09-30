import argparse
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import torch

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Filter out URL argument.')
    parser.add_argument(
        '--url',
        type=str,
        required=True,
        help='The URL to filter out.')
    args = parser.parse_args()
    url = args.url
    print(f"Using URL: {url}")

    # Load the web page content
    loader = WebBaseLoader(url)
    data = loader.load()

    # Split the loaded text into chunks for the vector database
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)
    print(f"Split into {len(all_splits)} chunks")

    # Embed the documents
    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )

    # Create a vector database using embeddings
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)

    # Print the number of documents
    print(f"Loaded {len(data)} documents")

    # Load the LlamaCpp LLM with the correct model path
    llm = LlamaCpp(
        model_path="/Users/paulstrebenitzer/repositories/HuggingFace/LLMs/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,
        temperature=0.5,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
    )
    print(f"Loaded LLM model from llama.cpp")

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever()
    )

    # Ask a question
    question = f"What are the latest headlines on {url}?"
    result = qa_chain.invoke({"query": question})
    print(result)

if __name__ == "__main__":
    main()
