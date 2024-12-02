from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize ChatOpenAI models
research_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
summary_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

# Initialize search tool
search_tool = DuckDuckGoSearchRun()

class ResearchState(TypedDict):
    query: str
    research_results: Annotated[List[str], operator.add]
    summary: str

graph = StateGraph(ResearchState)

def research_agent(state):
    query = state["query"]
    search_result = search_tool.run(query)
    return {"research_results": [search_result]}

def summarization_agent(state):
    research_results = state["research_results"]
    summary_prompt = f"Summarize the following research results:\n{research_results}"
    summary = summary_model.predict(summary_prompt)
    return {"summary": summary}

# Add nodes
graph.add_node("research", research_agent)
graph.add_node("summarize", summarization_agent)

# Set entry point and connect nodes
graph.set_entry_point("research")
graph.add_edge("research", "summarize")
graph.add_edge("summarize", END)

# Compile and run the graph
app = graph.compile()

def run_research(query):
    result = app.invoke({"query": query, "research_results": [], "summary": ""})
    return result["summary"]

# Example usage
research_topic = "Latest advancements in AI"
summary = run_research(research_topic)
print(f"Research Summary on '{research_topic}':\n{summary}")