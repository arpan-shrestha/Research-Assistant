# RAG-Based Research Assistant

A lightweight AI-powered research assistant that performs **retrieval-augmented generation (RAG)** over local documents using open-source tools. Ask natural language questions and get grounded answers with source references — all running **offline on a laptop**.
---
## Features

- Ingest `.pdf` documents into a **Chroma** vector database
- Fast **semantic search** using locally generated embeddings
- Natural-language Q&A using **Mistral 7B** (via Ollama)
- Short-term **session memory** for follow-up questions
- Automatic routing between **RAG** and **NL2SQL** for structured databases
- REST API with `/ask` and `/ingest` endpoints via **FastAPI**
---
## Project Structure
``` 
├── sample_docs/ # Folder for demo documents (.pdf)
├── chroma_db/ # Persisted local vector store (auto-created)
├── rag_pipeline.py # Core RAG logic, embeddings, memory, prompt
├── main.py # FastAPI app with /ask and /ingest endpoints
├── requirements.txt # All dependencies
├── architecture.png 
└── README.md 
```

---

## Setup Instructions

### 1. Clone and Install

```bash
git clone https://github.com/arpan-shrestha/Research-Assistant.git
cd rag-research-assistant
python -m venv env && source env/bin/activate
pip install -r requirements.txt
```
---
### 2. Start Ollama with Mistral
```bash
ollama serve
```
- Make sure that Ollama is installed and Mistral is pulled
---
### 3. Start the API server
```bash
uvicorn main:app --reload
```
---
### (Optional) 3b. Enable NL2SQL

Set the `SQL_DATABASE_URL` environment variable so the assistant can translate questions into SQL when necessary.

Examples:
```bash
export SQL_DATABASE_URL="sqlite:////absolute/path/to/my.db"
# or
export SQL_DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/mydb"
```
> The URL must be compatible with SQLAlchemy. Any required database drivers (e.g., `psycopg2-binary`, `pyodbc`) should also be installed in your environment.
---
### 4. Ingest documents (PDFs)
```bash
curl -X POST http://localhost:8000/ingest
```
---
### 5. Ask a question
```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Who is the captain that met Moby Dick?"}'
```
---
### Response
```json
{
  "answer": "The captain who encountered Moby Dick was Ahab...",
  "sources": ["sample_docs/Moby_Dick.pdf"],
  "mode": "rag",
  "metadata": {}
}
```
---

### System Architecture 
<img width="241" height="913" alt="Image" src="https://github.com/user-attachments/assets/c6f4096a-f67c-41ec-a50d-4ce263b252eb" />

---

### Video Link
https://drive.google.com/file/d/1KX12NqXU4zWtoibPLnS8PstunTGxydoM/view?usp=sharing
---
