from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_llm_cache
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
    
    prompt = "Hello world"
    
    # Run without caching
    print("Running without cache:")
    for i in range(1, 10):
        result = llm.invoke(prompt)
        print(f"Run {i}: {result}")
    
    # Run with caching
    print("\nRunning with cache:")
    for i in range(1, 10):
        result = llm.invoke(prompt)
        print(f"Run {i}: {result}")
    
    # Run with similar prompt
    print("\nRunning with similar prompt:")
    similar_prompt = "Hello world :)"
    for i in range(1, 10):
        result = llm.invoke(similar_prompt)
        print(f"Similar Run {i}: {result}")

# Run the demonstration
demonstrate_llm_caching()