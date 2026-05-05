from typing import Annotated, TypedDict, List, Union
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, ToolCall
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from src.config import Config
from src.tools import (
    search_agriculture_research,
    calculate_estimated_yield,
    duckduckgo_internet_search,
    send_expert_advice_email
)
from src.agent_config import (
    RESEARCHER_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    RESEARCHER_TOOLS,
    WRITER_TOOLS
)
import logging

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    Represents the state of our graph.
    - messages: full conversation history, merged by add_messages reducer
    - next: optional routing signal
    """
    messages: Annotated[list[BaseMessage], add_messages]
    next: str


def get_llm_with_tools(config: Config, system_prompt: str, tool_names: List[str]):
    """Helper to initialize LLM with specific tools and system prompt."""
    llm = ChatGroq(
        groq_api_key=config.GROQ_API_KEY,
        model_name=config.MODEL_NAME,
        temperature=0.1
    )

    available_tools = {
        "search_agriculture_research": search_agriculture_research,
        "calculate_estimated_yield":   calculate_estimated_yield,
        "duckduckgo_internet_search":  duckduckgo_internet_search,
        "send_expert_advice_email":    send_expert_advice_email
    }

    tools_to_bind = [available_tools[name] for name in tool_names if name in available_tools]
    return llm.bind_tools(tools_to_bind).with_config({"run_name": system_prompt.splitlines()[0]})


class MultiAgentAgricultureGraph:
    def __init__(self, checkpointer=None):
        self.config = Config()

        if checkpointer:
            self.memory = checkpointer
        else:
            # Persistence using SqliteSaver (Fixes KeyError: '__start__' in Streamlit)
            # We manually manage the connection to ensure it stays open for the life of the graph
            self.conn = sqlite3.connect(self.config.CHECKPOINT_DB_PATH, check_same_thread=False)
            self.memory = SqliteSaver(self.conn)

        # LLMs for each agent
        self.researcher_llm = get_llm_with_tools(self.config, RESEARCHER_SYSTEM_PROMPT, RESEARCHER_TOOLS)
        self.writer_llm     = get_llm_with_tools(self.config, WRITER_SYSTEM_PROMPT,     WRITER_TOOLS)

        self.workflow = StateGraph(AgentState)

        # -- Nodes --------------------------------------------------
        self.workflow.add_node("researcher", self._call_researcher)
        self.workflow.add_node("writer",     self._call_writer)

        available_tools = {
            "search_agriculture_research": search_agriculture_research,
            "calculate_estimated_yield":   calculate_estimated_yield,
            "duckduckgo_internet_search":  duckduckgo_internet_search,
            "send_expert_advice_email":    send_expert_advice_email
        }
        all_tool_names     = list(set(RESEARCHER_TOOLS + WRITER_TOOLS))
        all_tool_functions = [available_tools[name] for name in all_tool_names if name in available_tools]
        self.workflow.add_node("tool_node", ToolNode(all_tool_functions))

        # -- Entry point --------------------------------------------
        self.workflow.add_edge(START, "researcher")

        # -- Edges --------------------------------------------------
        self.workflow.add_conditional_edges(
            "researcher",
            self._route_researcher,
            {"tool_node": "tool_node", "writer": "writer", "end": END, "researcher": "researcher"}
        )
        self.workflow.add_edge("tool_node", "researcher")
        self.workflow.add_conditional_edges(
            "writer",
            self._route_writer,
            {"tool_node": "tool_node", "end": END}
        )

        # -- Compile ------------------------------------------------
        self.app = self.workflow.compile(
            checkpointer=self.memory,
            interrupt_before=["tool_node"]
        )

    # -- Agent node handlers ----------------------------------------

    def _call_researcher(self, state: AgentState) -> dict:
        messages = state['messages']
        new_messages = [HumanMessage(content=RESEARCHER_SYSTEM_PROMPT, name="system")] + [
            msg for msg in messages
            if not (isinstance(msg, HumanMessage) and msg.name == "system")
        ]
        response = self.researcher_llm.invoke(new_messages)
        logger.info(f"Researcher produced: {response}")
        return {"messages": [response]}

    def _call_writer(self, state: AgentState) -> dict:
        messages = state['messages']
        new_messages = [HumanMessage(content=WRITER_SYSTEM_PROMPT, name="system")] + [
            msg for msg in messages
            if not (isinstance(msg, HumanMessage) and msg.name == "system")
        ]
        response = self.writer_llm.invoke(new_messages)
        logger.info(f"Writer produced: {response}")
        return {"messages": [response]}

    # -- Routing logic ----------------------------------------------

    def _route_researcher(self, state: AgentState) -> str:
        last_message = state['messages'][-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tool_node"

        if isinstance(last_message, AIMessage) and "RESEARCH_COMPLETE" in last_message.content.upper():
            return "writer"

        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
            ai_messages_without_tools = [
                m for m in state['messages']
                if isinstance(m, AIMessage) and not m.tool_calls
            ]
            if len(ai_messages_without_tools) > 1 and "RESEARCH_COMPLETE" not in last_message.content.upper():
                return "end"
            return "researcher"

        return "researcher"

    def _route_writer(self, state: AgentState) -> str:
        last_message = state['messages'][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tool_node"
        return "end"


# -- Graph factory --------------------------------------------------
# If running inside Streamlit: use cache_resource so the graph
# survives hot-reloads and MemorySaver state stays intact.
# If running as plain Python (e.g. run_evaluation.py): instantiate
# directly without Streamlit dependency.

def _build_graph(use_checkpointer: bool = True, async_mode: bool = False):
    """Builds and returns a fresh compiled agriculture graph."""
    checkpointer = None
    if use_checkpointer and async_mode:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        # In async mode, we usually don't initialize the saver here 
        # but let the caller handle the connection lifespan.
        # However, for convenience:
        pass 

    manager = MultiAgentAgricultureGraph(checkpointer=None if not use_checkpointer else None)
    
    if not use_checkpointer:
        app = manager.workflow.compile()
        return app
    
    return manager.app


# Default instance with checkpointer for easy import
agriculture_graph = _build_graph()
