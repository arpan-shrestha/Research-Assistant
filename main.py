from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag_pipeline import load_and_split_documents, setup_chroma, load_chroma, query_chroma, generate_answer
from ingest import update_chroma  
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import os



USE_MOCK_LLM = os.getenv("MOCK_MODE") == "true"

app = FastAPI()

CHROMA_PATH = './chroma_db'

# Run once on startup
if not os.path.exists(CHROMA_PATH):
    docs = load_and_split_documents()
    setup_chroma(docs)
db = load_chroma()

# Store memory per session in a simple dict {session_id: memory_obj}
memory_store = {}

class QueryRequest(BaseModel):
    question: str
    session_id: str = None  # Add optional session_id

@app.post("/ask")
def ask_question(request: QueryRequest):
    # Use session_id or generate dummy one if none provided
    session_id = request.session_id or "default_session"
    
    # Get or create memory for this session
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(memory_key="chat_history")
    memory = memory_store[session_id]

    # Query chroma for relevant docs
    chunks = query_chroma(db, request.question)
    context = "\n\n---\n\n".join([doc.page_content for doc in chunks])

    # Build prompt with context, question, and conversation history
    PROMPT_TEMPLATE = """
    You are a helpful assistant. Use the conversation history and the following excerpts to answer the question.

    Conversation history:
    {chat_history}

    Excerpts:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        chat_history=memory.load_memory_variables({}).get("chat_history", ""),
        context=context,
        question=request.question
    )
    if USE_MOCK_LLM:
        answer = "This is a mock response for testing"
    else:
        model = OllamaLLM(model="mistral")
        answer = model.invoke(prompt)

    # Save this interaction to memory
    memory.save_context({"question": request.question}, {"answer": answer})

    sources = [doc.metadata.get("source", "unknown") for doc in chunks]
    
    print(f"[ASK] Session: {session_id}")
    print(f"[ASK] Question: {request.question}")
    print(f"[ASK] Answer (truncated): {answer[:2000]}...")

    return {
        "answer": answer,
        "sources": sources,
        "session_id": session_id 
    }

@app.post("/ingest")
def ingest():
    update_chroma()
    return {"message": "Chroma DB updated with new documents."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

