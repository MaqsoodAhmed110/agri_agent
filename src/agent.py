
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import json
import time
from typing import List, Dict
from src.config import Config
import logging

# Import the global metrics collector
from src.monitoring import metrics_collector as metrics

logger = logging.getLogger(__name__)

class AgricultureAgent:
    def __init__(self):
        self.config = Config()
        self.llm = ChatGroq(
            groq_api_key=self.config.GROQ_API_KEY,
            model_name=self.config.MODEL_NAME,
            temperature=0.1
        )
        # Use global metrics collector, don't create new one
        self.metrics = metrics
        
        self.system_prompt = """You are an expert AI assistant specialized in Agriculture research for Pakistan and South Asia. 
        Your knowledge comes exclusively from the provided research papers. 
        
        Guidelines:
        1. Answer based ONLY on the provided context from research papers
        2. If information is not in the context, say so - don't hallucinate
        3. Provide specific citations from the papers when possible
        4. Focus on practical agricultural implications for Pakistan/South Asia
        5. Structure your answers clearly with relevant examples
        
        Always cite sources using the paper titles provided in context."""
    
    def create_agent_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """Create formatted prompt with context"""
        context_str = "\n\n".join([
            f"Source: {chunk['metadata']['paper_title']} (Year: {chunk['metadata']['year']})\n"
            f"Content: {chunk['chunk']}"
            for chunk in context_chunks
        ])
        
        prompt = f"""SYSTEM: {self.system_prompt}

RESEARCH CONTEXT:
{context_str}

USER QUESTION: {query}

Please provide a comprehensive answer based on the research context above. Cite relevant sources and focus on practical implications for agriculture in Pakistan/South Asia."""
        
        return prompt
    
    def query(self, query: str, vector_store, k: int = 5) -> Dict:
        """Main agent query method"""
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant context
            context_chunks = vector_store.search(query, k=k)
            
            # Step 2: Create prompt
            prompt = self.create_agent_prompt(query, context_chunks)
            
            # Step 3: Generate response
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            # Step 4: Calculate metrics
            latency = (time.time() - start_time) * 1000  # ms
            
            # Log metrics using global collector
            self.metrics.log_query(
                query=query,
                response=response.content,
                context_chunks=context_chunks,
                latency=latency
            )
            
            return {
                'answer': response.content,
                'sources': context_chunks,
                'metrics': {
                    'latency_ms': latency,
                    'chunks_retrieved': len(context_chunks),
                    'unique_papers': len(set(chunk['metadata']['paper_title'] for chunk in context_chunks))
                }
            }
            
        except Exception as e:
            logger.error(f"Error in agent query: {e}")
            self.metrics.log_error()
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'metrics': {'latency_ms': (time.time() - start_time) * 1000, 'error': True}
            }