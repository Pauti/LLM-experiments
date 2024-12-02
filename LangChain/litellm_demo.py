import os
import streamlit as st
from dotenv import load_dotenv
from litellm import completion

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

def get_model_response(model_name: str, prompt: str) -> str:
    response = completion(model=model_name, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

st.title("Multi-Model Chat")

# Model selection
model_option = st.selectbox(
    "Choose a language model:",
    ("gpt-3.5-turbo", "ollama/llama2", "gpt-4o")
)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# User input
user_input = st.text_input("You:")

# Send button
if st.button("Send"):
    if openai_api_key or user_input:
        # Append user message to chat history
        st.session_state['chat_history'].append({"role": "user", "content": user_input})
        
        # Generate response using the selected model
        with st.spinner("Thinking..."):
            response = get_model_response(model_name=model_option, prompt=user_input)
        
        # Append model response to chat history
        st.session_state['chat_history'].append({"role": "model", "content": response})

# Display chat history in reverse order
for message in reversed(st.session_state['chat_history']):
    if message['role'] == 'user':
        st.write(f"You: {message['content']}")
    else:
        st.write(f"Model: {message['content']}")
