import faiss
import numpy as np
import pickle
import os
import logging
from typing import List, Dict

# Try multiple embedding options
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"SentenceTransformers not available: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    # Updated import path for HuggingFaceEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    LANGGCHAIN_EMBEDDINGS_AVAILABLE = True
except ImportError:
    try:
        # Fallback to old import path
        from langchain.embeddings import HuggingFaceEmbeddings
        LANGGCHAIN_EMBEDDINGS_AVAILABLE = True
    except ImportError:
        LANGGCHAIN_EMBEDDINGS_AVAILABLE = False

from src.config import Config

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.config = Config()
        self.embedding_model = self._initialize_embedding_model()
        self.index = None
        self.chunks = []
        self.metadata = []
        
    def _initialize_embedding_model(self):
        """Initialize embedding model with fallback options"""
        # Try sentence-transformers first
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Using SentenceTransformers with model: {self.config.EMBEDDING_MODEL}")
                model = SentenceTransformer(self.config.EMBEDDING_MODEL)
                # Test the model
                test_embedding = model.encode(["test"])
                logger.info(f"Model test successful. Embedding dimension: {test_embedding.shape[1]}")
                return model
            except Exception as e:
                logger.warning(f"SentenceTransformers failed: {e}")
        
        # Try LangChain HuggingFace embeddings with a known working model
        if LANGGCHAIN_EMBEDDINGS_AVAILABLE:
            try:
                logger.info("Using LangChain HuggingFace embeddings with all-MiniLM-L6-v2")
                model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                # Test the model
                test_embedding = model.embed_documents(["test"])
                logger.info(f"LangChain model test successful. Embedding dimension: {len(test_embedding[0])}")
                return model
            except Exception as e:
                logger.warning(f"LangChain embeddings failed: {e}")
        
        # Fallback to simple TF-IDF or other method
        raise ImportError("No suitable embedding model found. Please install sentence-transformers or langchain.")
    
    def create_embeddings(self, chunks: List[str], metadata: List[Dict]):
        """Create embeddings and build FAISS index"""
        logger.info("Creating embeddings...")
        
        if isinstance(self.embedding_model, SentenceTransformer):
            # Use sentence-transformers directly
            embeddings = self.embedding_model.encode(chunks)
            logger.info(f"SentenceTransformers embeddings shape: {embeddings.shape}")
        else:
            # Use LangChain embeddings
            embeddings_list = self.embedding_model.embed_documents(chunks)
            embeddings = np.array(embeddings_list, dtype='float32')
            logger.info(f"LangChain embeddings shape: {embeddings.shape}")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        logger.info(f"Creating FAISS index with dimension: {dimension}")
        
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize for cosine similarity
        logger.info("Normalizing embeddings...")
        faiss.normalize_L2(embeddings)
        
        # Add to index
        logger.info("Adding embeddings to index...")
        self.index.add(embeddings)
        
        self.chunks = chunks
        self.metadata = metadata
        
        logger.info(f"Created index with {len(chunks)} chunks")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        if self.index is None:
            raise ValueError("Index not initialized. Call create_embeddings first.")
            
        if isinstance(self.embedding_model, SentenceTransformer):
            query_embedding = self.embedding_model.encode([query]).astype('float32')
        else:
            query_embedding = np.array([self.embedding_model.embed_query(query)], dtype='float32')
            
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                results.append({
                    'chunk': self.chunks[idx],
                    'metadata': self.metadata[idx],
                    'score': float(distance),
                    'rank': i + 1
                })
        
        return results
    
    def save(self, path: str):
        """Save vector store to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, f"{path}.index")
        
        with open(f"{path}.metadata", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata
            }, f)
    
    def load(self, path: str):
        """Load vector store from disk"""
        self.index = faiss.read_index(f"{path}.index")
        
        with open(f"{path}.metadata", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']