# rag_pipeline.py
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
import os

CHROMA_PATH = './chroma_db'
DATA_PATH = 'sample_docs'

PROMPT_TEMPLATE = """
Use the following excerpts to answer the question:

{context}

Question: {question}
Answer:
"""

def load_and_split_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    return splitter.split_documents(documents)

# def setup_chroma(chunks):
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
#     db.add_documents(chunks)
#     return db
def setup_chroma(chunks, batch_size=5000):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        db.add_documents(batch)
    return db

def load_chroma():
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

def query_chroma(db, question, k=5):
    return db.similarity_search(question, k=k)

def generate_answer(chunks, question):
    context = "\n\n---\n\n".join([doc.page_content for doc in chunks])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context, question=question)
    model = OllamaLLM(model="mistral")
    return model.invoke(prompt)

