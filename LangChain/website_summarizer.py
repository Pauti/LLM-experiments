import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# load api key
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

def create_vectorstore(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(texts, embeddings)

def summarize_url(url):
    vectorstore = create_vectorstore(url)
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                     temperature=0.2,
                     api_key=openai_api_key)
    
    prompt_template = """Write a concise summary of the following text:

    {text}

    CONCISE SUMMARY:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = LLMChain(llm=llm, prompt=prompt)
    
    docs = vectorstore.similarity_search(query="", k=3)
    summaries = []
    for doc in docs:
        summary = chain.run(text=doc.page_content)
        summaries.append(summary)
    
    final_summary = "\n\n".join(summaries)
    return final_summary

# Lets go
url = input("Enter a URL to summarize: ")
summary = summarize_url(url)
print(f"SUMMARY OF {url}:\n{summary}")