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


with gd.Blocks(title="Research Assistant") as demo:
    gd.Markdown("# Research Assistant (RAG + SQL)")
    c = gd.Chatbot()
    m = gd.Textbox(label="Ask a question")
    m.submit(ask_question, [m, c], [c,m])

demo.launch()