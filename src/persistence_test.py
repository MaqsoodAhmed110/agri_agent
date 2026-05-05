import os
import sys
import uuid
from langchain_core.messages import HumanMessage
from src.multi_agent_graph import agriculture_graph
from src.config import Config # For checkpoint DB path

# Configure logging for better visibility
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Ensure the checkpoint DB exists (it will be created if not)
config = Config()
checkpoint_dir = os.path.dirname(config.CHECKPOINT_DB_PATH)
if checkpoint_dir and not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
logger.info(f"Using checkpoint DB at: {config.CHECKPOINT_DB_PATH}")

def run_conversation_segment(graph, thread_id: str, query: str):
    """Runs a single query segment for a given thread."""
    logger.info(f"\n--- Running query for thread_id: {thread_id} ---")
    logger.info(f"User: {query}")
    
    inputs = {"messages": [HumanMessage(content=query)]}
    
    # Use stream to show intermediate steps
    final_response = ""
    for s in graph.stream(inputs, config={"configurable": {"thread_id": thread_id}}):
        if "__end__" not in s:
            node_name = list(s.keys())[0]
            step_output = s[node_name]
            if 'messages' in step_output and step_output['messages']:
                last_msg = step_output['messages'][-1]
                if isinstance(last_msg, HumanMessage) and last_msg.name == "system":
                    pass # Skip system prompt messages for cleaner output
                elif hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    logger.info(f"  [{node_name}] ⚙️ Tool Call: {last_msg.tool_calls[0].name}({last_msg.tool_calls[0].args})")
                elif last_msg.type == 'tool':
                    logger.info(f"  [{node_name}] ✅ Tool Output: {last_msg.content[:100]}...")
                elif last_msg.type == 'ai':
                    logger.info(f"  [{node_name}] 🧠 Thought: {last_msg.content[:150]}...")
        else:
            final_messages = s["__end__"]["messages"]
            final_answer_message = None
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    final_answer_message = msg
                    break
                elif isinstance(msg, ToolMessage): # In case the last visible output is a tool message
                     final_answer_message = AIMessage(content=msg.content)
                     break
            
            if final_answer_message:
                final_response = final_answer_message.content
                logger.info(f"\nAgent Final Answer: {final_response}")
            else:
                logger.warning("\nAgent finished, but no clear final AIMessage was found.")
                # You might print the whole state for debugging
                logger.warning(f"Final State: {s['__end__']}")

    logger.info(f"--- Segment complete for thread_id: {thread_id} ---\n")
    return final_response

if __name__ == "__main__":
    # Ensure the agriculture_graph is loaded and compiled with checkpointer
    graph = agriculture_graph

    # Define a fixed thread_id for persistence demonstration
    test_thread_id = "test_persistence_thread_123"
    
    print(f"Starting persistence test with thread_id: '{test_thread_id}'")
    print("---------------------------------------------------------")

    # Segment 1: Initial query
    logger.info("--- Phase 1: Initial Query ---")
    run_conversation_segment(graph, test_thread_id, "What are the common challenges for wheat cultivation in Punjab, Pakistan?")

    logger.info("--- Phase 2: Follow-up query (should remember context) ---")
    logger.info("--- Simulating script restart (re-run persistence_test.py) ---")
    input("Press Enter to run the next query (simulating script restart)...")

    # Segment 2: Follow-up query, should leverage previous context
    run_conversation_segment(graph, test_thread_id, "And what about the recommended fertilizers for those challenges?")

    logger.info("--- Phase 3: A new, unrelated query for the same thread_id (should still remember previous context) ---")
    input("Press Enter to run the final query (should still remember previous context)...")

    # Segment 3: Another query, still on the same thread
    run_conversation_segment(graph, test_thread_id, "What are the typical rice yields in Sindh?")

    print("\nPersistence Test Complete.")
    print(f"You can inspect '{config.CHECKPOINT_DB_PATH}' to see the saved states.")
    print("Try running this script again with the same thread_id and observe the continuity.")
