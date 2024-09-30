# Import the Ollama class from the langchain_community.llms module
from langchain_community.llms import Ollama

# These classes are used for handling the output of the language model
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Create an instance of the Ollama language model
# The 'temperature' argument is set to 0.9, which controls the randomness of the output
llm = Ollama(model="llama2",
             # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
             temperature=0.9)

# This class is used to define the input and output structure of the language model
from langchain.prompts import PromptTemplate

# Define a PromptTemplate
# The output is expected to be a list of five facts related to the 'topic'
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me 5 facts about {topic}.",
)

# This class is used to create a chain of language models
from langchain.chains import LLMChain

# Create an instance of the LLMChain class
chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

# Run the language model chain
print(chain.invoke("the moon"))

