from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from src.vector_store import VectorStore
import os
import logging

# Ensure langchain_community is installed: pip install langchain-community duckduckgo-search
from langchain_community.tools import DuckDuckGoSearchRun

logger = logging.getLogger(__name__)

class SearchInput(BaseModel):
    query: str = Field(description="The specific search query to look up in agriculture research papers")
    k: int = Field(default=5, description="Number of relevant research chunks to retrieve")

class YieldInput(BaseModel):
    crop: str = Field(description="The name of the crop (e.g., wheat, rice, cotton)")
    area_acres: float = Field(description="The land area in acres")
    region: str = Field(description="The region in Pakistan/South Asia (e.g., Punjab, Sindh)")

class DuckDuckGoSearchInput(BaseModel):
    query: str = Field(description="The search query to perform on the internet using DuckDuckGo.")

class EmailInput(BaseModel):
    recipient: str = Field(description="The email address of the recipient.")
    subject: str = Field(description="The subject line of the email.")
    body: str = Field(description="The full content body of the email.")

# Global vector store instance for tools
_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        try:
            _vector_store = VectorStore()
            index_path = os.path.join("data", "embeddings", "agriculture_index")
            if os.path.exists(index_path + ".index"):
                _vector_store.load(index_path)
            else:
                logger.warning(f"Vector store index not found at {index_path}. Search tool may not function.")
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {e}")
            _vector_store = None # Ensure it's None if initialization fails
    return _vector_store

@tool("search_agriculture_research", args_schema=SearchInput)
def search_agriculture_research(query: str, k: int = 5) -> str:
    """Queries the local vector database containing agricultural research papers for Pakistan and South Asia. 
    Use this tool to find factual information, citations, and research-backed data."""
    vs = get_vector_store()
    if not vs or not vs.index:
        return "Error: Research database not initialized or loaded."
    
    results = vs.search(query, k=k)
    context = []
    if not results:
        return "No relevant agricultural research found in the local database."
    
    for res in results:
        meta = res.get('metadata', {})
        title = meta.get('paper_title', 'Unknown Title')
        year = meta.get('year', 'Unknown Year')
        content = res.get('chunk', 'No content')
        context.append(f"Source: {title} (Year: {year})\nContent: {content}")
    
    return "\n\n---\n\n".join(context)

@tool("calculate_estimated_yield", args_schema=YieldInput)
def calculate_estimated_yield(crop: str, area_acres: float, region: str) -> str:
    """Calculates estimated crop yield based on regional averages in Pakistan.
    Useful for practical implications and planning."""
    # Placeholder logic for regional yield averages (kg per acre)
    yield_map = {
        "wheat": {"punjab": 1200, "sindh": 1100, "balochistan": 900, "kpk": 1000, "default": 1000},
        "rice": {"punjab": 950, "sindh": 1050, "balochistan": 800, "kpk": 900, "default": 900},
        "cotton": {"punjab": 700, "sindh": 800, "balochistan": 600, "kpk": 650, "default": 700}
    }
    
    crop_data = yield_map.get(crop.lower(), None)
    
    if not crop_data:
        return f"Warning: No specific yield data for crop '{crop}'. Cannot estimate yield."

    avg_yield = crop_data.get(region.lower(), crop_data["default"])
    total = avg_yield * area_acres
    
    return f"Based on regional data for {region}, the estimated yield for {area_acres} acres of {crop} is {total:.2f} kg."

@tool("duckduckgo_internet_search", args_schema=DuckDuckGoSearchInput)
def duckduckgo_internet_search(query: str) -> str:
    """Performs a general internet search using DuckDuckGo to find information beyond the local research papers.
    Use this for broader or more current information that might not be in the agricultural research database.
    Always prefer the 'search_agriculture_research' tool for specific, research-backed agricultural data."""
    search = DuckDuckGoSearchRun()
    try:
        results = search.run(query)
        if not results:
            return "No relevant information found on the internet."
        return results
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return f"Error performing internet search: {e}"

@tool("send_expert_advice_email", args_schema=EmailInput)
def send_expert_advice_email(recipient: str, subject: str, body: str) -> str:
    """Sends an email with agricultural advice or reports to a specified recipient.
    This is a high-risk action requiring human approval."""
    # For demonstration, we'll just print to console.
    # In a real application, this would integrate with an email sending service (e.g., SMTP, SendGrid).
    email_content = f"""
    --- SIMULATED EMAIL SENT ---
    To: {recipient}
    Subject: {subject}
    Body:
    {body}
    ---------------------------
    """
    logger.info(f"Simulating email send: {email_content}")
    return f"Email successfully prepared for sending to {recipient} with subject '{subject}'. (Simulation)"
