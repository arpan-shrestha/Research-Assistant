# from langchain_ollama import OllamaEmbeddings

# def get_embedding_function():
#     embedding = OllamaEmbeddings(model="nomic-embed-text")
#     return embedding
import os

USE_MOCK_LLM = os.getenv("MOCK_MODE") == "true"

class MockEmbeddings:
    def embed_documents(self, texts):
        # Return a list of zero vectors of length 768 (or your desired embedding size)
        return [[0.0] * 768 for _ in texts]

    def embed_query(self, text):
        # Return zero vector for query
        return [0.0] * 768

def get_embedding_function():
    if USE_MOCK_LLM:
        return MockEmbeddings()
    else:
        from langchain_ollama import OllamaEmbeddings
        embedding = OllamaEmbeddings(model="nomic-embed-text")
        return embedding
