from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_openai import OpenAI
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

def demonstrate_llm_caching():
    # Set up the LLM
    llm = OpenAI(model="gpt-3.5-turbo-instruct", n=2, best_of=2)
    
    # Enable in-memory caching
    set_llm_cache(InMemoryCache())
    
    # First invocation (not cached)
    start_time = time.time()
    result1 = llm.invoke("Tell me a joke")
    end_time = time.time()
    print(f"First invocation (not cached):")
    print(f"Result: {result1}")
    print(f"Time taken: {end_time - start_time:.4f} seconds\n")
    
    # Second invocation (cached)
    start_time = time.time()
    result2 = llm.invoke("Tell me a joke")
    end_time = time.time()
    print(f"Second invocation (cached):")
    print(f"Result: {result2}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")


# Run the demonstration
demonstrate_llm_caching()