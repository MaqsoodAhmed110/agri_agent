import sys
import os
import uuid
from langchain_core.messages import HumanMessage

# Add project root to path
sys.path.append(os.path.abspath(os.getcwd()))

from src.multi_agent_graph import _build_graph

def test_keyerror():
    print("Building graph with checkpointer...")
    graph = _build_graph(use_checkpointer=True)
    
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {"messages": [HumanMessage(content="Hello", name="user")]}
    
    print(f"Starting stream with thread_id: {thread_id}")
    try:
        for s in graph.stream(inputs, config=config, stream_mode="values"):
            print(f"Step: {list(s.keys()) if isinstance(s, dict) else type(s)}")
        print("Success!")
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_keyerror()
