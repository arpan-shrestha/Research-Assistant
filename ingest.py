# ingest.py
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
import os
import pandas as pd

Chroma_Path = './chroma_db'
Data_Path = 'sample_docs'

def load_pdf():
    loader = PyPDFDirectoryLoader(Data_Path)
    return loader.load()

def load_csv():
    csv_docs=[]
    for file in os.listdir(Data_Path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(Data_Path, file))
            for _, row in df.iterrows():
                text = " ".join(str(x) for x in row.values)
                csv_docs.append(Document(page_content=text, metadata={"source": file}))
    return csv_docs

def load_docs():
    pdf_docs = load_pdf()
    csv_docs = load_csv()
    return pdf_docs + csv_docs

def split_docs(docs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len
    )
    return splitter.split_documents(docs)

def update_chroma():
    print("Loading new documents...")
    docs = load_docs()
    chunks = split_docs(docs)

    print("Updating Chroma DB...")
    db = Chroma(
        persist_directory=Chroma_Path,
        embedding_function=get_embedding_function()
    )

    db.add_documents(chunks)
    print("Chroma DB updated with new documents.")

if __name__ == "__main__":
    update_chroma()
