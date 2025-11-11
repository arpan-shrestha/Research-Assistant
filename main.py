from fastapi import FastAPI
from pydantic import BaseModel, validator
from rag_pipeline import load_and_split_documents, setup_chroma, load_chroma, query_chroma, generate_answer
from ingest import update_chroma  
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
import os
import asyncio
import hashlib
from collections import OrderedDict
from threading import Lock
from datetime import datetime, timedelta

USE_MOCK_LLM = os.getenv("MOCK_MODE") == "true"

app = FastAPI()

CHROMA_PATH = './chroma_db'

# Global variables initialized on startup
llm_model = None
db = None
prompt_template = None

# Store memory per session in a simple dict {session_id: memory_obj}
memory_store = {}
memory_store_lock = Lock()  # Thread safety for memory_store
memory_last_access = {}  # Track last access time for cleanup
MAX_SESSION_AGE_HOURS = 24  # Clean up sessions older than 24 hours
MAX_SESSIONS = 1000  # Maximum number of sessions to keep

# Query result cache (LRU cache using OrderedDict)
query_cache = OrderedDict()
query_cache_lock = Lock()  # Thread safety for query_cache
CACHE_SIZE = 100  # Limit cache size

def get_cache_key(question: str, session_id: str) -> str:
    """Generate a cache key for a query"""
    return hashlib.md5(f"{session_id}:{question}".encode()).hexdigest()

def get_from_cache(cache_key: str):
    """Get item from cache and move to end (LRU) - thread-safe"""
    with query_cache_lock:
        if cache_key in query_cache:
            # Move to end (most recently used)
            query_cache.move_to_end(cache_key)
            return query_cache[cache_key]
    return None

def add_to_cache(cache_key: str, value: dict):
    """Add item to cache with LRU eviction - thread-safe"""
    with query_cache_lock:
        if cache_key in query_cache:
            # Update existing and move to end
            query_cache.move_to_end(cache_key)
            query_cache[cache_key] = value
        else:
            # Add new item
            query_cache[cache_key] = value
            # Evict oldest if cache is full
            if len(query_cache) > CACHE_SIZE:
                query_cache.popitem(last=False)  # Remove oldest (first) item

def cleanup_old_sessions():
    """Remove old sessions to prevent memory leaks"""
    with memory_store_lock:
        now = datetime.now()
        sessions_to_remove = []
        
        for session_id, last_access in list(memory_last_access.items()):
            age = now - last_access
            if age > timedelta(hours=MAX_SESSION_AGE_HOURS):
                sessions_to_remove.append(session_id)
        
        # If still too many sessions, remove oldest ones
        if len(memory_store) > MAX_SESSIONS:
            # Sort by last access and remove oldest
            sorted_sessions = sorted(memory_last_access.items(), key=lambda x: x[1])
            excess = len(memory_store) - MAX_SESSIONS
            for session_id, _ in sorted_sessions[:excess]:
                if session_id not in sessions_to_remove:
                    sessions_to_remove.append(session_id)
        
        # Remove old sessions
        for session_id in sessions_to_remove:
            memory_store.pop(session_id, None)
            memory_last_access.pop(session_id, None)
        
        if sessions_to_remove:
            print(f"[CLEANUP] Removed {len(sessions_to_remove)} old sessions")

# Initialize everything on startup (async to avoid blocking)
@app.on_event("startup")
async def startup_event():
    global llm_model, db, prompt_template
    
    print("[STARTUP] Initializing application...")
    
    # Initialize LLM
    if not USE_MOCK_LLM:
        print("[STARTUP] Loading LLM model...")
        llm_model = OllamaLLM(model="mistral")
        print("[STARTUP] LLM model loaded")
    
    # Initialize Chroma DB (run in thread pool to avoid blocking)
    print("[STARTUP] Loading Chroma DB...")
    if not os.path.exists(CHROMA_PATH):
        print("[STARTUP] Chroma DB not found, creating new database...")
        # Load and split documents in thread pool
        docs = await asyncio.to_thread(load_and_split_documents)
        # Setup Chroma in thread pool
        await asyncio.to_thread(setup_chroma, docs)
        print("[STARTUP] Chroma DB created")
    
    # Load Chroma DB in thread pool
    db = await asyncio.to_thread(load_chroma)
    print("[STARTUP] Chroma DB loaded")
    
    # Pre-warm embedding function (test with a dummy query)
    print("[STARTUP] Pre-warming embedding function...")
    try:
        embedding_func = get_embedding_function()
        await asyncio.to_thread(embedding_func.embed_query, "test")
        print("[STARTUP] Embedding function ready")
    except Exception as e:
        print(f"[STARTUP] Warning: Could not pre-warm embeddings: {e}")
    
    # Create and cache prompt template (do this once)
    PROMPT_TEMPLATE = """
    You are a helpful assistant. Use the conversation history and the following excerpts to answer the question.
    
    Conversation History:
    {chat_history}
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    print("[STARTUP] Prompt template cached")
    
    print("[STARTUP] Application ready!")

class QueryRequest(BaseModel):
    question: str
    session_id: str = None  # Add optional session_id
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        if len(v) > 10000:  # Reasonable limit
            raise ValueError("Question is too long (max 10000 characters)")
        return v.strip()

@app.post("/ask")
async def ask_question(request: QueryRequest):
    global db, prompt_template
    
    # Check if DB is initialized
    if db is None:
        raise RuntimeError("Chroma DB not initialized. Please wait for startup to complete.")
    
    # Cleanup old sessions periodically (every 100 requests, roughly)
    if len(memory_store) % 100 == 0:
        cleanup_old_sessions()
    
    # Use session_id or generate dummy one if none provided
    session_id = request.session_id or "default_session"
    
    # Check cache first
    cache_key = get_cache_key(request.question, session_id)
    cached_response = get_from_cache(cache_key)
    if cached_response is not None:
        print(f"[ASK] Cache hit for question: {request.question[:50]}...")
        return cached_response
    
    # Get or create memory for this session (thread-safe)
    with memory_store_lock:
        if session_id not in memory_store:
            memory_store[session_id] = ConversationBufferMemory(memory_key="chat_history")
        memory = memory_store[session_id]
        memory_last_access[session_id] = datetime.now()

    # Load memory variables in thread pool to avoid blocking
    memory_vars = await asyncio.to_thread(memory.load_memory_variables, {})
    chat_history = memory_vars.get("chat_history", "")

    # Query chroma for relevant docs (run in thread pool to avoid blocking)
    try:
        chunks = await asyncio.to_thread(query_chroma, db, request.question)
        if not chunks:
            raise ValueError("No relevant documents found in the database. Please ensure documents are ingested.")
        context = "\n\n---\n\n".join([doc.page_content for doc in chunks])
    except Exception as e:
        print(f"[ASK] Error querying Chroma DB: {e}")
        raise RuntimeError(f"Failed to query database: {str(e)}")

    # Build prompt using cached template
    prompt = prompt_template.format(
        chat_history=chat_history,
        context=context,
        question=request.question
    )
    
    # Use async execution for LLM call to avoid blocking
    try:
        if USE_MOCK_LLM:
            answer = "This is a mock response for testing"
        else:
            if llm_model is None:
                raise RuntimeError("LLM model not initialized. Please check Ollama is running.")
            # Run LLM invoke in thread pool to avoid blocking the event loop
            answer = await asyncio.to_thread(llm_model.invoke, prompt)
            if not answer or not answer.strip():
                raise RuntimeError("LLM returned an empty response")
    except Exception as e:
        print(f"[ASK] Error calling LLM: {e}")
        raise RuntimeError(f"Failed to generate answer: {str(e)}")

    # Save this interaction to memory (run in thread pool)
    await asyncio.to_thread(memory.save_context, {"question": request.question}, {"answer": answer})

    sources = [doc.metadata.get("source", "unknown") for doc in chunks]
    
    response = {
        "answer": answer,
        "sources": sources,
        "session_id": session_id 
    }
    
    # Cache the response (with LRU eviction)
    add_to_cache(cache_key, response)
    
    print(f"[ASK] Session: {session_id}")
    print(f"[ASK] Question: {request.question}")
    print(f"[ASK] Answer (truncated): {answer[:200]}...")

    return response

@app.post("/ingest")
async def ingest():
    global db
    # Run ingestion in thread pool to avoid blocking
    try:
        await asyncio.to_thread(update_chroma)
        # Reload the DB to include new documents
        print("[INGEST] Reloading Chroma DB to include new documents...")
        db = await asyncio.to_thread(load_chroma)
        print("[INGEST] Chroma DB reloaded successfully")
        return {"message": "Chroma DB updated with new documents and reloaded."}
    except Exception as e:
        print(f"[INGEST] Error during ingestion: {e}")
        raise RuntimeError(f"Failed to ingest documents: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}
