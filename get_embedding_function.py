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

# Cache the embedding function to avoid recreating it
_embedding_function = None

def get_embedding_function():
<<<<<<< HEAD
    if os.getenv("MOCK_MODE") == "true":
        return MockEmbeddings()
    else:
        embedding = OllamaEmbeddings(model="nomic-embed-text")
        return embedding
=======
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    return _embedding_function
>>>>>>> 4301fef (requirements.txt)
