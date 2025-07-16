from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    return embedding
