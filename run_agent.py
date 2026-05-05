#!/usr/bin/env python3
"""
Main script to run the Agriculture Research Agent
"""

import os
import sys
from src.data_processor import DataProcessor
from src.vector_store import VectorStore
from src.agent import AgricultureAgent

def setup_data_pipeline():
    """Set up the complete data processing pipeline"""
    print("🚀 Setting up Agriculture Research Agent...")
    
    # Process PDFs and create chunks
    processor = DataProcessor()
    print("📄 Processing research papers...")
    processed_data = processor.process_all_papers("data/metadata.csv")
    
    # Create and save embeddings
    vector_store = VectorStore()
    print("🔧 Creating embeddings...")
    vector_store.create_embeddings(
        processed_data['chunks'], 
        processed_data['metadata']
    )
    
    # Save vector store
    vector_store.save("data/embeddings/agriculture_index")
    print("✅ Data pipeline setup complete!")
    
    return vector_store

if __name__ == "__main__":
    # Check if vector store exists
    if not os.path.exists("data/embeddings/agriculture_index.index"):
        print("First-time setup required...")
        setup_data_pipeline()
    
    print("🎯 Starting Agriculture Research Agent...")
    print("📊 Metrics available at: http://localhost:8000")
    print("🌐 Web interface available at: http://localhost:8501")
    print("📈 Grafana dashboard at: http://localhost:3001 (admin/admin)")
    
    # Start the Streamlit app
    os.system("streamlit run app/main.py")