# RAG Chatbot – Chat with PDF

A terminal-based Retrieval-Augmented Generation (RAG) app that lets you ask questions about any PDF, powered by **Google Gemini**.

---

## Features

- 📄 Extracts text from any PDF (including large multi-page documents)  
- ✂️ Splits text into overlapping chunks (`2000` chars, `250` char overlap) for optimized retrieval  
- 🔢 Embeds every chunk using Google's `gemini-embedding-001`  
- ⏳ **Free Tier Rate-Limit Resilience:** Automatic 429 / `RESOURCE_EXHAUSTED` retry with dynamic wait times, plus automatic pauses between batches to stay within the 100 RPM limit  
- 🔍 Retrieves the most relevant chunks via cosine similarity  
- 🤖 Answers your questions using `gemini-3.5-flash-lite` with grounded document context  
- 💬 Maintains multi-turn conversation history  

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your Gemini API key

```bash
# Copy the example env file
copy .env.example .env
```

Then open `.env` and replace `your_gemini_api_key_here` with your real key.  
Get one free at <https://aistudio.google.com/app/apikey>.

---

## Usage

### Option 1: Modern Web Interface (Streamlit) 🌐
```bash
streamlit run app.py
```
This opens an interactive ChatGPT-like web UI in your browser where you can drag & drop your PDF and chat directly.

### Option 2: Terminal / CLI Interface 💻
```bash
python rag_chat.py path/to/your/document.pdf
```

### In-chat commands

| Command  | Description                        |
|----------|------------------------------------|
| `/help`  | Show available commands            |
| `/clear` | Clear conversation history         |
| `/quit`  | Exit the application (or `Ctrl+C`) |

---

## Architecture

```
PDF file
   │
   ▼
pypdf  ──► full text
   │
   ▼
chunk_text()  ──► overlapping character chunks  (2000 chars, 250 overlap)
   │
   ▼
client.models.embed_content()  ──► numpy embeddings matrix  (gemini-embedding-001)
   │                                [auto-retry on 429 & batch pacing]
   ▼
 On each user query:
   │
   ├─► embed query  ──► cosine similarity  ──► top-5 chunks
   │
   └─► build grounded prompt + history  ──► gemini-3.5-flash-lite  ──► answer
```

---

## Project Structure

```
RAG Chatbot AI/
├── rag_chat.py        # Main application (RAG pipeline + terminal chat)
├── requirements.txt   # Python dependencies (google-genai, pypdf, numpy, python-dotenv)
├── .env.example       # API key template
├── .env               # Your actual API key (keep private)
└── README.md          # Documentation
```

---

## Tips & Configurations

- **Large PDFs (e.g., 100+ pages):** With chunk size set to `2000` characters, PDFs produce fewer chunks. If total chunks exceed 100, the script automatically batches them and pauses 60s between batches to respect Gemini Free Tier's 100 requests-per-minute quota.
- **Parameters:** You can customize `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K` directly at the top of [`rag_chat.py`](file:///d:/Projects/RAG%20Chatbot%20AI/rag_chat.py).
- **History reset:** Use `/clear` to reset chat context between unrelated questions.
- **Exiting:** Type `/quit`, `/exit`, `/q`, or press `Ctrl + C`.
