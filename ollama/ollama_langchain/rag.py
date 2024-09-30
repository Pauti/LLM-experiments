# Load web page
import argparse # command line arguments

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embed and store
from langchain.vectorstores import Chroma # vector database
from langchain.embeddings import GPT4AllEmbeddings # GPT-4 embeddings
from langchain.embeddings import OllamaEmbeddings # Ollama embeddings

from langchain_community.llms import Ollama # Ollama LLM
from langchain.callbacks.manager import CallbackManager # handles callbacks from Langchain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # Callback Handler for Streaming

# Main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Filter out URL argument.')
    parser.add_argument(
        '--url',
        type=str,
        default='http://example.com',
        required=True,
        help='The URL to filter out.')

    # Get the arguments and print the URL
    args = parser.parse_args()
    url = args.url
    print(f"using URL: {url}")

    # Load the web page content
    loader = WebBaseLoader(url)
    data = loader.load()

    # Split the loaded text into chunks for the vector database
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)
    print(f"Split into {len(all_splits)} chunks")

    # Create a vector database using embeddings
    vectorstore = Chroma.from_documents(documents=all_splits,
                                        embedding=OllamaEmbeddings())

    # Print the number of documents
    print(f"Loaded {len(data)} documents")

    # Fetch the prompt template from the langchain hub
    from langchain import hub
    QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama") # https://smith.langchain.com/hub/rlm/rag-prompt-llama

    # Load the Ollama LLM
    llm = Ollama(model="llama2",
                verbose=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    print(f"Loaded LLM model {llm.model}")

    # Create the QA chain
    # This chain first does a retrieval step to fetch relevant documents, 
    # then passes those documents into an LLM to generate a response.
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(), # use the vector database containing all the chunks
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    # Ask a question
    question = f"What are the latest headlines on {url}?"
    result = qa_chain({"query": question})

    # Print the result
    # print(result)
    
if __name__ == "__main__":
    main()