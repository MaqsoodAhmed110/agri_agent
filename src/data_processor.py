import fitz  # PyMuPDF
import pandas as pd
import os
from typing import List, Dict
import logging
from src.config import Config  # Changed this line

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.config = Config()
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks"""
        chunk_size = chunk_size or self.config.CHUNK_SIZE
        overlap = overlap or self.config.CHUNK_OVERLAP
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
            
            if end >= len(text):
                break
                
        return chunks
    
    def process_all_papers(self, metadata_file: str) -> Dict:
        """Process all PDFs and create chunks"""
        metadata = pd.read_csv(metadata_file)
        all_chunks = []
        chunk_metadata = []
        
        for _, row in metadata.iterrows():
            pdf_filename = row['file_name']  # Assuming your CSV has file_name column
            pdf_path = os.path.join(self.config.RAW_PDFS, pdf_filename)
            
            if os.path.exists(pdf_path):
                text = self.extract_text_from_pdf(pdf_path)
                chunks = self.chunk_text(text)
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    chunk_metadata.append({
                        'paper_id': row.get('id', _),
                        'paper_title': row['title'],
                        'authors': row.get('authors', ''),
                        'year': row.get('year', ''),
                        'chunk_id': i,
                        'source_file': pdf_filename
                    })
                
                logger.info(f"Processed {pdf_filename}: {len(chunks)} chunks")
            else:
                logger.warning(f"PDF not found: {pdf_path}")
        
        return {
            'chunks': all_chunks,
            'metadata': chunk_metadata
        }