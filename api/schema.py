from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user message to send to the agent")
    thread_id: str = Field(..., description="Unique session ID for conversation persistence")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The final response from the agent")
    status: str = Field(default="completed", description="The current status of the request")
    thread_id: str = Field(..., description="The session ID associated with this response")

class StreamChunk(BaseModel):
    node: str = Field(..., description="The name of the node that just executed")
    content: Optional[str] = Field(None, description="Partial content or status update")
    status: str = Field(default="processing")
