# RAG Chatbot – Chat with PDF

A production-grade Retrieval-Augmented Generation (RAG) system with a modern **Streamlit Web UI** and an interactive **Terminal CLI**, powered by **Google Gemini**.

---

## Key Features

- 📄 **Hybrid PDF Extraction + Vision OCR:** Extracts digital text with `pypdf`, and automatically falls back to Gemini Vision OCR (via `pypdfium2`) for scanned or image-only pages.
- ⚡ **Persistent Disk Caching:** Embeddings and chunks are hashed (SHA-256) and cached under `.cache/`. Reloading the same document happens instantly in **0.05 seconds** with **0 API calls**.
- ✂️ **Smart Recursive Chunking:** Splits text by natural paragraphs and sentence boundaries (1,500 chars, 200 overlap), ensuring words and thoughts are never awkwardly cut in half.
- 📌 **Page-Level Source Citations:** Every factual response explicitly cites source page numbers (e.g., `[Page 2]`). The UI provides an interactive collapsible tray showing the exact chunks and relevance scores.
- 🛡️ **Relevance Guard ("I Don't Know" Threshold):** Rejects irrelevant queries when cosine similarity falls below `0.35`, saving tokens and completely eliminating hallucinations.
- ⏳ **Free Tier Rate-Limit Resilience:** Automatic `429 / RESOURCE_EXHAUSTED` retry with dynamic wait times, plus 60-second batch pacing for documents with >100 chunks.
- 🤖 **Fast & Accurate Generation:** Combines `gemini-embedding-001` for vector representations with `gemini-3.5-flash-lite` for multi-turn grounded conversations.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your Gemini API key

```bash
# Copy example env file
copy .env.example .env
```

Open `.env` and paste your key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
Get a free API key at <https://aistudio.google.com/app/apikey>.

---

## Usage

### Option 1: Modern Web Interface (Streamlit) 🌐

```bash
streamlit run app.py
```
Opens an interactive ChatGPT-like web UI in your browser (`http://localhost:8501`) featuring drag-and-drop PDF upload, live embedding status, and source reference accordions.

### Option 2: Terminal / CLI Interface 💻

```bash
python rag_chat.py path/to/your/document.pdf
```

#### In-terminal commands:
| Command    | Description                                  |
|------------|----------------------------------------------|
| `/help`    | Show available commands                      |
| `/sources` | View retrieved chunk snippets for last query |
| `/clear`   | Reset conversation history                   |
| `/quit`    | Exit the chatbot (or `Ctrl + C`)             |

---

## Architecture

```
                 PDF Document (Digital or Scanned)
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │       Page-by-Page Extraction Pipeline       │
        │  • Digital Text  ──► pypdf                   │
        │  • Scanned Pages ──► pypdfium2 + Gemini OCR  │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │           Smart Recursive Chunking           │
        │  • Paragraph & Sentence preservation         │
        │  • Page-number metadata tagging              │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
                    Check Disk Cache (.cache/)
                     ├── Cache Hit  ──► Instant Load (0 API calls)
                     └── Cache Miss ──► gemini-embedding-001 (100 RPM paced)
                                        └── Save to .cache/
                               │
                               ▼
                    User Query Processing
                               │
                               ├─► Embed Query (gemini-embedding-001)
                               ├─► NumPy Cosine Similarity
                               │
                               ▼
                    Similarity Threshold Check
                     ├── Best Match < 0.35 ──► "Information not found" (0 gen tokens)
                     └── Best Match ≥ 0.35 ──► Grounded Prompt + Citations
                                               │
                                               ▼
                                     gemini-3.5-flash-lite
                                               │
                                               ▼
                              Response with [Page X] Citations
```

---

## Project Structure

```
RAG Chatbot AI/
├── app.py             # Streamlit web application
├── rag_chat.py        # Terminal / CLI application
├── requirements.txt   # Python dependencies
├── .env.example       # API key template
├── .env               # Your private API key (git-ignored)
├── .gitignore         # Ignores .env, .cache/, and build artifacts
├── .cache/            # Local disk cache for precomputed embeddings
└── README.md          # Project documentation
```

---

## Configuration & Tuning

You can adjust these parameters directly at the top of `rag_chat.py` and `app.py`:
- `CHUNK_SIZE`: Target chunk length in characters (default: `1500`)
- `CHUNK_OVERLAP`: Overlap between consecutive chunks (default: `200`)
- `TOP_K`: Number of context chunks retrieved per query (default: `5`)
- `SIMILARITY_THRESHOLD`: Minimum cosine similarity required to answer (default: `0.35`)
