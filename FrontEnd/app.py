import gradio as gd
import requests

ask_url = "http://127.0.0.1:8000/ask"
ingest_url = "http://127.0.0.1:8000/ingest"

def ask_question(msg, hist):
    payload = {"question":msg}
    response = requests.post(ask_url, json=payload)
    answer = response.json().get("answer", "Error")
    hist = hist + [
        {"role":"user", "content": msg},
        {"role": "assistant", "content":answer},
    ]
    return hist, ""

def ingest():
    res = requests.post(ingest_url)
    return res.json()

def clear_chat():
    return []


with gd.Blocks(title="Research Assistant") as demo:
    gd.Markdown("# Research Assistant (RAG + SQL)")
    c = gd.Chatbot()
    m = gd.Textbox(label="Ask a question")
    m.submit(ask_question, [m, c], [c,m])
    
    with gd.Row():
        ingest_btn = gd.Button ("File Ingest")
        clear_btn = gd.Button("Clear Chat")
    status = gd.Textbox(label="Status", interactive=False)

    ingest_btn.click(ingest, None, status)
    clear_btn.click(clear_chat, None, c)

demo.launch()