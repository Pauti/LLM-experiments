from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# load api key
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize the LLM
#llm = Ollama(model="llama2")
llm = ChatOpenAI(model="gpt-3.5-turbo", 
             api_key=openai_api_key)

# Set up the conversation memory
memory = ConversationBufferMemory()

# Create the conversation chain
conversation = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=True
)

# Main chat loop
while True:
    user_input = input("Human: ")
    if user_input.lower() == 'quit':
        break
    response = conversation.predict(input=user_input)
    print("AI:", response)