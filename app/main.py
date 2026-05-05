import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import time
import uuid
from typing import Union
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from src.monitoring import metrics_collector as metrics
from src.tools import send_expert_advice_email, EmailInput
from src.multi_agent_graph import _build_graph
graph = _build_graph(use_checkpointer=False)  # no checkpointer for eval
# Define constants for HITL flow
PROCEED_ACTION      = "Proceed"
CANCEL_ACTION       = "Cancel"
EDIT_PROCEED_ACTION = "Edit & Proceed"


# ── Graph cached so MemorySaver survives Streamlit hot-reloads ────
@st.cache_resource
def get_graph_and_memory():
    from src.multi_agent_graph import _build_graph
    return _build_graph()


def get_session_id():
    """
    Returns current session display ID (shown in sidebar).
    Actual thread_id used for LangGraph is generated fresh per query
    to avoid KeyError: '__start__' in LangGraph 1.x MemorySaver.
    """
    if 'graph_init_id' not in st.session_state:
        st.session_state.graph_init_id     = str(uuid.uuid4())
        st.session_state.session_id        = str(uuid.uuid4())
        st.session_state.current_thread_id = str(uuid.uuid4())
        st.session_state.run_id            = str(uuid.uuid4())
        st.session_state.messages          = []
        st.session_state.interrupted_state = None
    return st.session_state.session_id


def _get_stable_thread_id() -> str:
    """
    Returns the STABLE thread_id for the current session.
    Enables true persistence across multiple queries.
    """
    if 'current_thread_id' not in st.session_state:
        st.session_state.current_thread_id = str(uuid.uuid4())
    return st.session_state.current_thread_id


def main():
    st.set_page_config(
        page_title="Agri-Agent (Persistent & HITL)",
        page_icon="🌾",
        layout="wide"
    )

    st.title("🌾 Autonomous Agri-Agent with Persistence & Human-in-the-Loop")
    st.markdown("""
    This agent uses a **ReAct (Reason + Act)** loop with multiple specialists. It features:
    - **Persistent Memory**: Remembers conversations across sessions using a `thread_id`.
    - **Human-in-the-Loop (HITL)**: Pauses for approval before high-risk actions (e.g., sending an email).
    - **State Editing**: Allows humans to modify proposed actions before approval.
    """)

    # ── Sidebar ───────────────────────────────────────────────────
    with st.sidebar:
        st.header("System Monitoring")
        st.info("Agent Architecture: LangGraph (ReAct Multi-Agent)")

        st.subheader("Session & Persistence")
        session_id = get_session_id()
        st.write(f"Current Session ID: `{session_id}`")
        st.write(f"Active Thread ID: `{st.session_state.get('current_thread_id', 'N/A')}`")
        st.caption("This thread ID persists across queries to maintain conversation memory.")

        if st.button("Start New Conversation"):
            st.session_state.session_id        = str(uuid.uuid4())
            st.session_state.current_thread_id = str(uuid.uuid4())
            st.session_state.messages          = []
            st.session_state.interrupted_state = None
            st.session_state.run_id            = str(uuid.uuid4())
            st.rerun()

        if st.button("Clear Current History"):
            st.session_state.messages          = []
            st.session_state.interrupted_state = None
            st.rerun()

    # ── Get graph (cached) ────────────────────────────────────────
    graph = get_graph_and_memory()

    # ── Init remaining session state ──────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "interrupted_state" not in st.session_state:
        st.session_state.interrupted_state = None
    if "run_id" not in st.session_state:
        st.session_state.run_id = str(uuid.uuid4())
    if "current_thread_id" not in st.session_state:
        st.session_state.current_thread_id = str(uuid.uuid4())

    # ── Display chat history ──────────────────────────────────────
    for message in st.session_state.messages:
        role = message["role"] if message["role"] != "tool_output" else "assistant"
        with st.chat_message(role):
            if isinstance(message["content"], list):
                for item in message["content"]:
                    st.json(item) if isinstance(item, dict) else st.markdown(str(item))
            else:
                st.markdown(message["content"])

    # ── HITL interruption UI ──────────────────────────────────────
    if st.session_state.interrupted_state:
        st.warning("❗ Agent requires human approval for a high-risk action!")

        interrupted_messages = st.session_state.interrupted_state.get('messages', [])
        last_agent_message   = interrupted_messages[-1] if interrupted_messages else None

        if last_agent_message and hasattr(last_agent_message, 'tool_calls') and last_agent_message.tool_calls:
            proposed_tool_call = last_agent_message.tool_calls[0]

            if proposed_tool_call['name'] == "send_expert_advice_email":
                st.subheader("Proposed Email Action:")

                default_recipient = proposed_tool_call['args'].get('recipient', '')
                default_subject   = proposed_tool_call['args'].get('subject', '')
                default_body      = proposed_tool_call['args'].get('body', '')

                edited_recipient = st.text_input("Recipient", value=default_recipient, key="hitl_recipient")
                edited_subject   = st.text_input("Subject",   value=default_subject,   key="hitl_subject")
                edited_body      = st.text_area("Body",       value=default_body, height=200, key="hitl_body")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(PROCEED_ACTION, key="hitl_proceed"):
                        st.session_state.interrupted_state = None
                        _resume_agent_execution(last_agent_message)
                with col2:
                    if st.button(EDIT_PROCEED_ACTION, key="hitl_edit_proceed"):
                        try:
                            edited_args = EmailInput(
                                recipient=edited_recipient,
                                subject=edited_subject,
                                body=edited_body
                            ).dict()
                            edited_tool_call = {
                                'name': proposed_tool_call['name'],
                                'args': edited_args,
                                'id':   proposed_tool_call['id']
                            }
                            st.session_state.interrupted_state = None
                            _resume_agent_execution(edited_tool_call)
                        except Exception as e:
                            st.error(f"Invalid email edits: {e}")
                with col3:
                    if st.button(CANCEL_ACTION, key="hitl_cancel"):
                        st.session_state.interrupted_state = None
                        cancel_message = HumanMessage(
                            content=f"Human cancelled the '{proposed_tool_call['name']}' action.",
                            name="human"
                        )
                        _resume_agent_execution(cancel_message)

            else:
                st.write(f"Proposed Tool Call: `{proposed_tool_call['name']}` with args: `{proposed_tool_call['args']}`")
                if st.button(PROCEED_ACTION, key="hitl_proceed_other"):
                    st.session_state.interrupted_state = None
                    _resume_agent_execution(last_agent_message)
                if st.button(CANCEL_ACTION, key="hitl_cancel_other"):
                    st.session_state.interrupted_state = None
                    cancel_message = HumanMessage(
                        content=f"Human cancelled the '{proposed_tool_call['name']}' action.",
                        name="human"
                    )
                    _resume_agent_execution(cancel_message)

    # ── Chat input — disabled during HITL ─────────────────────────
    if not st.session_state.interrupted_state:
        if query := st.chat_input("Ask about Pakistan's agriculture (e.g., 'What are wheat yield estimates for 50 acres in Punjab?')"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            _run_agent_query(graph, query)


def _run_agent_query(graph, query: str):
    """
    Executes the agent graph with a FRESH thread_id per query.
    Consistent with run_evaluation.py approach.

    No session_id parameter — thread is managed internally via
    _get_fresh_thread_id() to fix KeyError: '__start__' in LangGraph 1.x.
    """
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        start_time    = time.time()

        # Use stable thread_id — enabled by SqliteSaver fix
        thread_id = _get_stable_thread_id()
        config    = {"configurable": {"thread_id": thread_id}}

        try:
            inputs = {"messages": [HumanMessage(content=query, name="user")]}

            for s in graph.stream(inputs, config=config, stream_mode="values"):
                messages = s.get("messages", [])
                if not messages:
                    continue

                last_msg = messages[-1]

                if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                    tc = last_msg.tool_calls[0]
                    tool_name = tc['name'] if isinstance(tc, dict) else tc.name
                    full_response = f"⚙️ **Calling Tool:** `{tool_name}`\n\n*Waiting for human approval...*"
                    message_placeholder.markdown(full_response)
                    st.session_state.interrupted_state = s
                    st.rerun()

                elif isinstance(last_msg, ToolMessage):
                    full_response += f"✅ **Tool Result:** {last_msg.content[:300]}\n\n"
                    message_placeholder.markdown(full_response + "▌")

                elif isinstance(last_msg, AIMessage) and last_msg.content:
                    full_response = last_msg.content
                    message_placeholder.markdown(full_response)

            # Finalise if no interruption occurred
            if not st.session_state.interrupted_state:
                if full_response:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    full_response = "Agent finished processing."
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

            latency = (time.time() - start_time) * 1000
            metrics.log_query(
                query=query,
                response=full_response,
                context_chunks=[],
                latency=latency
            )

        except Exception as e:
            err = str(e)
            message_placeholder.error(f"Error during agent execution: {err}")
            metrics.log_error()
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}"})


def _resume_agent_execution(action_message):
    """
    Resumes after HITL using the current_thread_id stored in session state
    so the resume targets the same checkpoint that was interrupted.
    No session_id parameter — consistent with _run_agent_query approach.
    """
    graph     = get_graph_and_memory()
    thread_id = st.session_state.get("current_thread_id", str(uuid.uuid4()))
    config    = {"configurable": {"thread_id": thread_id}}

    if isinstance(action_message, dict) and 'name' in action_message and 'args' in action_message:
        tool_map = {"send_expert_advice_email": send_expert_advice_email}
        tool_fn  = tool_map.get(action_message['name'])
        try:
            tool_output_str = tool_fn(**action_message['args']) if tool_fn else f"Unknown tool: {action_message['name']}"
        except Exception as e:
            tool_output_str = f"Tool execution error: {str(e)}"

        message_to_inject = ToolMessage(
            content=str(tool_output_str),
            tool_call_id=action_message['id'],
            name=action_message['name']
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"✅ Tool `{action_message['name']}` executed after human approval."
        })

    elif isinstance(action_message, HumanMessage):
        message_to_inject = action_message
        st.session_state.messages.append({"role": "assistant", "content": action_message.content})

    else:
        message_to_inject = None

    try:
        result = graph.invoke(
            {"messages": [message_to_inject]} if message_to_inject else None,
            config=config
        )

        final_response = "Action completed."
        if result and "messages" in result:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    final_response = msg.content
                    break

        st.session_state.messages.append({"role": "assistant", "content": final_response})

    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"Resume error: {str(e)}"})

    st.rerun()


if __name__ == "__main__":
    main()