# from langchain_ollama import OllamaEmbeddings

# def get_embedding_function():
#     embedding = OllamaEmbeddings(model="nomic-embed-text")
#     return embedding

from langchain_ollama import OllamaEmbeddings
from langchain.embeddings.base import Embeddings
import os
import numpy as np

class MockEmbeddings(Embeddings):
    """Mock embeddings for testing without Ollama"""
    
    def embed_documents(self, texts):
        # Return random embeddings for testing
        return [np.random.rand(384).tolist() for _ in texts]
    
    def embed_query(self, text):
        # Return random embedding for testing
        return np.random.rand(384).tolist()

def get_embedding_function():
    if os.getenv("MOCK_MODE") == "true":
        return MockEmbeddings()
    else:
        embedding = OllamaEmbeddings(model="nomic-embed-text")
        return embedding
