from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

with get_openai_callback() as callback:
  llm.invoke("What is AI?")

print(callback)