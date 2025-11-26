# ingest.py
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
import os
import pandas as pd

Chroma_Path = './chroma_db'
Data_Path = 'sample_docs'

def load_pdf():
    """Load PDF files from directory"""
    try:
        loader = PyPDFDirectoryLoader(Data_Path)
        docs = loader.load()
        print(f"Loaded {len(docs)} PDF documents")
        return docs
    except Exception as e:
        print(f"Error loading PDF files: {e}")
        raise

def load_csv():
    """Load CSV files and convert to documents - optimized with vectorized operations"""
    csv_docs = []
    try:
        for file in os.listdir(Data_Path):
            if file.endswith('.csv'):
                file_path = os.path.join(Data_Path, file)
                try:
                    df = pd.read_csv(file_path)
                    # Use vectorized operations instead of iterrows (much faster)
                    # Convert all rows to strings at once
                    df_str = df.astype(str)
                    # Join all columns for each row
                    texts = df_str.apply(lambda row: " ".join(row.values), axis=1)
                    # Create documents in batch
                    for text in texts:
                        csv_docs.append(Document(page_content=text, metadata={"source": file}))
                    print(f"Loaded {len(texts)} rows from {file}")
                except Exception as e:
                    print(f"Error loading CSV file {file}: {e}")
                    continue
    except Exception as e:
        print(f"Error accessing data directory: {e}")
        raise
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
    """Update Chroma DB with new documents - safe ingestion"""
    try:
        print("Loading new documents...")
        docs = load_docs()
        if not docs:
            print("Warning: No documents found to ingest")
            return

        print(f"Splitting {len(docs)} documents into chunks...")
        chunks = split_docs(docs)
        print(f"Created {len(chunks)} chunks")

        print("Updating Chroma DB...")
        db = Chroma(
            persist_directory=Chroma_Path,
            embedding_function=get_embedding_function()
        )

        batch_size = 50  # smaller batches reduce Ollama crashes
        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                db.add_documents(batch)
                total_added += len(batch)
                if total_added % 100 == 0 or i + batch_size >= len(chunks):
                    print(f"Added {total_added} / {len(chunks)} chunks...")
            except Exception as e:
                print(f"[WARNING] Failed to embed batch {i}-{i+batch_size}: {e}")
                continue  # skip batch if embedding fails

        print(f"Chroma DB updated with {total_added} chunks from {len(docs)} documents.")

    except Exception as e:
        print(f"[ERROR] Updating Chroma DB failed: {e}")
        raise

if __name__ == "__main__":
    update_chroma()
