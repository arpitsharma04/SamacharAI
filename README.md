# 📰 SamacharAI — Multi-Language News Research Tool

Ask questions from any news article and get answers in **English and Hindi** — built with LangChain, Ollama, FAISS, and Streamlit. Completely free, no API key needed.

> Built while learning LangChain and GenAI. AI tools were used as a reference during development.

---

## Features

- Paste up to 3 news article URLs
- Ask in English or Hindi
- Get side-by-side answers in English 🇬🇧 and Hindi 🇮🇳
- Runs 100% locally using Ollama — no cost, no data sent online
- Clean dark UI

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Streamlit | Web UI |
| LangChain | RAG pipeline |
| Ollama llama3.2:3b | Local LLM |
| Ollama nomic-embed-text | Local embeddings |
| FAISS | Vector store |
| UnstructuredURLLoader | Load news from URLs |

---

## Setup

### 1. Install Ollama and pull models
Download from [https://ollama.com](https://ollama.com)

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama serve
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
streamlit run main.py
```

---

## Project Structure

```
SamacharAI/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

Built by Arpit Sharma — BTech student 🚀
