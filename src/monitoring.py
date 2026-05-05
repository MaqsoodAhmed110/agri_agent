from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
import json
import time
import logging
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

# Create metrics once at module level (not in class init)
_metrics_initialized = False
_query_counter = None
_error_counter = None
_latency_histogram = None
_chunks_retrieved = None
_unique_papers_gauge = None

def init_metrics_once():
    """Initialize metrics only once to avoid duplication"""
    global _metrics_initialized, _query_counter, _error_counter, _latency_histogram, _chunks_retrieved, _unique_papers_gauge
    
    if not _metrics_initialized:
        try:
            # Clear existing metrics with same names
            for name in list(REGISTRY._names_to_collectors.keys()):
                if name.startswith('agent_'):
                    REGISTRY.unregister(REGISTRY._names_to_collectors[name])
            
            # Create metrics
            _query_counter = Counter('agent_queries_total', 'Total number of queries')
            _error_counter = Counter('agent_errors_total', 'Total number of errors')
            _latency_histogram = Histogram('agent_latency_ms', 'Query latency in milliseconds')
            _chunks_retrieved = Histogram('chunks_retrieved_count', 'Number of chunks retrieved per query')
            _unique_papers_gauge = Gauge('unique_papers_retrieved', 'Unique papers in context')
            
            _metrics_initialized = True
            logger.info("Metrics initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing metrics: {e}")

class MetricsCollector:
    def __init__(self, port=8000):
        self.port = port
        
        # Initialize metrics once
        init_metrics_once()
        
        # Start metrics server in background thread
        try:
            # Check if server is already running
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', self.port))
            sock.close()
            
            if result != 0:  # Port not in use
                start_http_server(self.port)
                logger.info(f"Metrics server started on port {self.port}")
            else:
                logger.info(f"Metrics server already running on port {self.port}")
        except Exception as e:
            logger.error(f"Error starting metrics server: {e}")
    
    def log_query(self, query: str, response: str, context_chunks: list, latency: float):
        """Log query metrics"""
        if not _metrics_initialized:
            init_metrics_once()
            
        try:
            _query_counter.inc()
            _latency_histogram.observe(latency)
            _chunks_retrieved.observe(len(context_chunks))
            
            unique_papers = len(set(chunk['metadata']['paper_title'] for chunk in context_chunks))
            _unique_papers_gauge.set(unique_papers)
            
            # Log to file
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'query': query[:100],
                'response_length': len(response),
                'chunks_retrieved': len(context_chunks),
                'unique_papers': unique_papers,
                'latency_ms': latency
            }
            
            with open('logs/metrics.log', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Error logging query: {e}")
    
    def log_error(self):
        """Log error metrics"""
        if not _metrics_initialized:
            init_metrics_once()
            
        try:
            _error_counter.inc()
        except Exception as e:
            logger.error(f"Error logging error: {e}")

# Create global instance
metrics_collector = MetricsCollector()