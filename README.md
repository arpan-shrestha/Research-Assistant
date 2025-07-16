# RAG-Based Research Assistant

A lightweight AI-powered research assistant that performs **retrieval-augmented generation (RAG)** over local documents using open-source tools. Ask natural language questions and get grounded answers with source references — all running **offline on a laptop**.
---
## 🚀 Features

- 📄 Ingest `.pdf` documents into a **Chroma** vector database
- 🔍 Fast **semantic search** using locally generated embeddings
- 🤖 Natural-language Q&A using **Mistral 7B** (via Ollama)
- 🧠 Short-term **session memory** for follow-up questions
- ⚡ REST API with `/ask` and `/ingest` endpoints via **FastAPI**
- 🧪 Unit tested with `pytest`
- 📦 CI-ready with **GitHub Actions**
---
## 📂 Project Structure
``` 
├── sample_docs/ # Folder for demo documents (.pdf)
├── chroma_db/ # Persisted local vector store (auto-created)
├── rag_pipeline.py # Core RAG logic, embeddings, memory, prompt
├── main.py # FastAPI app with /ask and /ingest endpoints
├── tests/
│ └── test_rag.py 
├── requirements.txt # All dependencies
├── architecture.png 
└── README.md 
```

---

## Setup Instructions

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/rag-research-assistant.git
cd rag-research-assistant
python -m venv env && source env/bin/activate
pip install -r requirements.txt
```
---
### 2. Start Ollama with Mistral
```bash
ollama run mistral
```
Make sure that **Ollama** is installed and **Mistral** is pulled
---
### 3. Start the API server
```bash
uvicorn main:app --reload
```
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
  "sources": ["sample_docs/Moby_Dick.pdf"]
}
```
---

### System Architecture 
<img width="241" height="913" alt="Image" src="https://github.com/user-attachments/assets/c6f4096a-f67c-41ec-a50d-4ce263b252eb" />
