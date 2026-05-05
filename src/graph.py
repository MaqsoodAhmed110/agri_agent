from typing import Annotated, TypedDict, Union
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from src.config import Config
from src.tools import search_agriculture_research, calculate_estimated_yield

class AgentState(TypedDict):
    # The add_messages function handles appending new messages to the history
    messages: Annotated[list, add_messages]

def create_agriculture_graph():
    config = Config()
    llm = ChatGroq(
        groq_api_key=config.GROQ_API_KEY,
        model_name=config.MODEL_NAME,
        temperature=0.1
    )
    
    # Define the toolset
    tools = [search_agriculture_research, calculate_estimated_yield]
    llm_with_tools = llm.bind_tools(tools)

    # Node 1: The Agent
    def call_model(state: AgentState):
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Build the Graph
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    # Set Edges
    workflow.add_edge(START, "agent")
    
    # Conditional Router
    workflow.add_conditional_edges(
        "agent",
        tools_condition, # Prebuilt router: checks if LLM wants to call tools
    )
    
    # Edge from tools back to agent to process tool results
    workflow.add_edge("tools", "agent")

    return workflow.compile()

# Example usage interface
graph = create_agriculture_graph()