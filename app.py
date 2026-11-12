"""
RAG Chatbot – Web Interface (Streamlit)
========================================
Interactive ChatGPT-like web UI for chatting with your PDF documents.
Powered by Google Gemini:
  • Embedding Model: gemini-embedding-001
  • Chat Model:      gemini-3.5-flash-lite
Features:
  • Persistent Disk Cache (instant 0-second reloads)
  • Exact Page Number Citations & Source Accordion
  • Smart Paragraph & Sentence Chunking
  • Relevance Guard ("I don't know" threshold to stop hallucinations)
  • Hybrid Digital Text + Vision OCR (for scanned PDFs)
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import re
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pypdfium2 as pdfium
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Chatbot AI - Chat with PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
TOP_K = 5
SIMILARITY_THRESHOLD = 0.35

CACHE_DIR = Path(".cache")

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a precise, grounded AI document assistant that answers questions
    STRICTLY using the provided context snippets.

    CRITICAL RULES:
    1. Grounding: Answer ONLY based on the facts directly stated in the context.
       Do not assume, extrapolate, or use outside knowledge.
    2. Citations: Every single factual claim, date, or detail in your answer MUST
       cite its source page number using the format [Page X] (e.g., "Sagar completed
       his M.A. in Clinical Psychology [Page 2]").
    3. Honesty: If the provided context does NOT contain enough information to
       answer the question, state clearly: "The provided document does not contain
       information to answer this question." Do not try to guess.
""")


# ---------------------------------------------------------------------------
# Data Models & Caching
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    chunk_id: int
    page: int
    text: str


def compute_bytes_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of file content."""
    return hashlib.sha256(file_bytes).hexdigest()


def get_cache_path(file_hash: str) -> Path:
    """Return path to cached embeddings."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{file_hash}.pkl"


def load_cached_embeddings(file_hash: str) -> Optional[Tuple[List[DocumentChunk], np.ndarray]]:
    """Load pre-computed chunks and embeddings from disk if available."""
    cache_file = get_cache_path(file_hash)
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        if data.get("embed_model") != EMBED_MODEL:
            return None
        chunks = [
            DocumentChunk(chunk_id=c["chunk_id"], page=c["page"], text=c["text"])
            for c in data["chunks"]
        ]
        embeddings = np.array(data["embeddings"], dtype=np.float32)
        return chunks, embeddings
    except Exception:
        return None


def save_cached_embeddings(
    file_hash: str,
    chunks: List[DocumentChunk],
    embeddings: np.ndarray,
) -> None:
    """Save chunks and embeddings to disk cache."""
    cache_file = get_cache_path(file_hash)
    data = {
        "embed_model": EMBED_MODEL,
        "chunks": [asdict(c) for c in chunks],
        "embeddings": embeddings,
    }
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"[Warning] Failed to write cache: {e}")


# ---------------------------------------------------------------------------
# PDF Extraction & OCR Fallback
# ---------------------------------------------------------------------------

def ocr_page_image(client: genai.Client, image_bytes: bytes) -> str:
    """Use Gemini Vision to OCR a scanned PDF page image."""
    try:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        prompt = (
            "Transcribe all text from this page image verbatim. "
            "Maintain original paragraphs and layout. Return only the extracted text."
        )
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=[prompt, image_part],
        )
        return response.text.strip()
    except Exception as e:
        print(f"Vision OCR error: {e}")
        return ""


def load_pdf_pages_from_bytes(
    file_bytes: bytes,
    client: genai.Client,
    status_box=None,
) -> List[Tuple[int, str]]:
    """Extract page text from PDF bytes with OCR fallback for scanned pages."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: List[Tuple[int, str]] = []
    pdfium_doc = None

    for i, page in enumerate(reader.pages, start=1):
        raw_text = (page.extract_text() or "").strip()
        if len(raw_text) >= 30:
            pages.append((i, raw_text))
            continue

        # Scanned page detected
        if status_box:
            status_box.write(f"🔍 Page {i} is an image/scanned. Running Gemini Vision OCR...")

        try:
            if pdfium_doc is None:
                pdfium_doc = pdfium.PdfDocument(file_bytes)
            pdf_page = pdfium_doc[i - 1]
            pil_image = pdf_page.render(scale=2.0).to_pil()

            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            ocr_text = ocr_page_image(client, buf.getvalue())

            if ocr_text:
                pages.append((i, ocr_text))
            if status_box:
                status_box.write(f"✅ Page {i}: OCR extracted {len(ocr_text)} characters.")
        except Exception as e:
            if status_box:
                status_box.write(f"⚠️ Page {i} OCR skipped: {e}")

    if not pages:
        raise ValueError("No extractable text or images found in the uploaded PDF.")
    return pages


# ---------------------------------------------------------------------------
# Smart Chunking
# ---------------------------------------------------------------------------

def split_text_smart(
    text: str,
    max_chars: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split text recursively by paragraphs and sentences without cutting words."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        if len(para) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(current_chunk) + len(sent) + 1 <= max_chars:
                    current_chunk = f"{current_chunk} {sent}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                        current_chunk = f"{overlap_text} {sent}".strip()
                    else:
                        chunks.append(sent[:max_chars])
                        current_chunk = sent[max_chars - overlap:]
        else:
            if len(current_chunk) + len(para) + 2 <= max_chars:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = f"{overlap_text}\n\n{para}".strip()
                else:
                    chunks.append(para)
                    current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def build_document_chunks(pages: List[Tuple[int, str]]) -> List[DocumentChunk]:
    """Build smart chunks tagged with exact page numbers."""
    all_chunks: List[DocumentChunk] = []
    chunk_counter = 1

    for page_num, page_text in pages:
        page_chunks = split_text_smart(page_text, max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for text_chunk in page_chunks:
            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_counter,
                    page=page_num,
                    text=text_chunk,
                )
            )
            chunk_counter += 1

    return all_chunks


# ---------------------------------------------------------------------------
# Embeddings & Retrieval
# ---------------------------------------------------------------------------

def embed_content_with_retry(
    client: genai.Client,
    model: str,
    contents: str | list[str],
    config: types.EmbedContentConfig,
    status_placeholder=None,
) -> types.EmbedContentResponse:
    """Wrapper around client.models.embed_content with automatic 429 retry."""
    while True:
        try:
            return client.models.embed_content(
                model=model,
                contents=contents,
                config=config,
            )
        except (errors.ClientError, errors.APIError) as exc:
            err_msg = str(exc)
            if getattr(exc, "code", None) == 429 or "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                wait_sec = 60.0
                match = re.search(
                    r"(?:retryDelay|retry in|retry after)[\"':\s]*([0-9]+(?:\.[0-9]+)?)\s*s?",
                    err_msg,
                    re.IGNORECASE,
                )
                if match:
                    try:
                        wait_sec = float(match.group(1))
                    except ValueError:
                        wait_sec = 60.0

                total_wait = wait_sec + 2.0
                msg = f"⏳ Free Tier limit reached (429). Waiting {total_wait:.0f}s before retrying..."
                if status_placeholder:
                    status_placeholder.warning(msg)
                time.sleep(total_wait)
            else:
                raise


def embed_chunks(client: genai.Client, chunks: List[DocumentChunk], progress_callback=None) -> np.ndarray:
    """Embed chunks in batches of 100 with Free Tier rate-limit protection."""
    texts = [c.text for c in chunks]
    batch_size = 100
    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_idx = (i // batch_size) + 1
        batch = texts[i : i + batch_size]

        if progress_callback:
            progress_callback(batch_idx, total_batches, len(batch))

        response = embed_content_with_retry(
            client=client,
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        for emb in response.embeddings:
            all_embeddings.append(emb.values)

        if batch_idx < total_batches:
            if progress_callback:
                progress_callback(batch_idx, total_batches, len(batch), is_pausing=True)
            time.sleep(60.0)

    return np.array(all_embeddings, dtype=np.float32)


def embed_query(client: genai.Client, query: str) -> np.ndarray:
    """Embed single user query."""
    response = embed_content_with_retry(
        client=client,
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return np.array(response.embeddings[0].values, dtype=np.float32)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity calculation."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return d @ q


def retrieve(
    client: genai.Client,
    query: str,
    chunks: List[DocumentChunk],
    chunk_embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> Tuple[List[DocumentChunk], List[float], float]:
    """Retrieve top-K chunks with individual scores and max score."""
    q_vec = embed_query(client, query)
    scores = cosine_similarity(q_vec, chunk_embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved = [chunks[i] for i in top_indices]
    top_scores = [float(scores[i]) for i in top_indices]
    max_score = top_scores[0] if top_scores else 0.0

    return retrieved, top_scores, max_score


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_history" not in st.session_state:
    st.session_state.api_history = []

if "indexed_pdf_hash" not in st.session_state:
    st.session_state.indexed_pdf_hash = None

if "indexed_pdf_name" not in st.session_state:
    st.session_state.indexed_pdf_name = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = None


@st.cache_resource
def get_gemini_client(api_key: str):
    if not api_key:
        return None
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta"),
    )


client = get_gemini_client(GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📄 RAG Chatbot AI")
    st.caption("Powered by Google Gemini & Streamlit")
    st.markdown("---")

    if not GEMINI_API_KEY:
        st.error("⚠️ `GEMINI_API_KEY` not found in `.env`")
        user_key = st.text_input("Enter Gemini API Key:", type="password")
        if user_key:
            client = get_gemini_client(user_key)
            GEMINI_API_KEY = user_key
    else:
        st.success("🔑 API Key loaded from `.env`", icon="✅")

    st.markdown("### 📥 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None and client is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = compute_bytes_hash(file_bytes)

        # Check if already in session
        if st.session_state.indexed_pdf_hash != file_hash:
            # Check persistent disk cache first
            cached_data = load_cached_embeddings(file_hash)

            if cached_data is not None:
                chunks, chunk_embeddings = cached_data
                st.session_state.chunks = chunks
                st.session_state.chunk_embeddings = chunk_embeddings
                st.session_state.indexed_pdf_hash = file_hash
                st.session_state.indexed_pdf_name = uploaded_file.name
                st.session_state.messages = []
                st.session_state.api_history = []
                st.toast(f"⚡ Loaded {len(chunks)} chunks from disk cache (0 API calls)!", icon="🚀")
                st.rerun()
            else:
                status_box = st.status(f"Processing '{uploaded_file.name}'...", expanded=True)
                try:
                    status_box.write("📖 Extracting pages (digital text + OCR fallback)...")
                    pages = load_pdf_pages_from_bytes(file_bytes, client, status_box=status_box)

                    status_box.write("✂️ Smart-chunking into paragraphs & sentences...")
                    chunks = build_document_chunks(pages)
                    status_box.write(f"Created {len(chunks)} semantic chunks.")

                    def update_progress(batch_num, total_batches, batch_len, is_pausing=False):
                        if is_pausing:
                            status_box.write(f"⏳ Free Tier limit: Pausing 60s before Batch {batch_num + 1}...")
                        else:
                            status_box.write(f"🔢 Embedding Batch {batch_num}/{total_batches} ({batch_len} chunks)...")

                    chunk_embeddings = embed_chunks(client, chunks, progress_callback=update_progress)

                    # Save to persistent disk cache
                    status_box.write("💾 Saving embeddings to disk cache (.cache/)...")
                    save_cached_embeddings(file_hash, chunks, chunk_embeddings)

                    # Store in session state
                    st.session_state.chunks = chunks
                    st.session_state.chunk_embeddings = chunk_embeddings
                    st.session_state.indexed_pdf_hash = file_hash
                    st.session_state.indexed_pdf_name = uploaded_file.name
                    st.session_state.messages = []
                    st.session_state.api_history = []

                    status_box.update(label="✅ PDF Indexed & Cached Successfully!", state="complete", expanded=False)
                    st.rerun()

                except Exception as e:
                    status_box.update(label="❌ Error processing PDF", state="error", expanded=True)
                    st.error(f"Error: {e}")

    # Document details if indexed
    if st.session_state.indexed_pdf_name:
        st.markdown("---")
        st.markdown("### 📊 Document Info")
        st.info(
            f"**File:** `{st.session_state.indexed_pdf_name}`\n\n"
            f"**Total Chunks:** `{len(st.session_state.chunks)}`\n\n"
            f"**Disk Cache:** `⚡ Enabled (.cache/)`\n\n"
            f"**Embedding:** `{EMBED_MODEL}`\n\n"
            f"**Chat Model:** `{CHAT_MODEL}`"
        )

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.api_history = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color: gray;'>Built with Streamlit & Google GenAI SDK</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------

st.header("💬 Chat with your PDF", divider="rainbow")

if not st.session_state.indexed_pdf_name:
    st.info("👈 **To get started:** Upload a PDF file from the sidebar on the left.")
    st.markdown(
        """
        ### What this app does:
        1. 📄 **Reads your PDF** (both digital and scanned image pages via Vision OCR)
        2. ⚡ **Saves embeddings to disk cache** (instant 0-second reloads in future)
        3. 🔍 **Finds exact paragraphs** using `gemini-embedding-001` with sentence preservation
        4. 🛡️ **Guards against hallucinations** using confidence thresholding
        5. 📌 **Cites exact page numbers** (`[Page X]`) for every fact!
        """
    )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📚 View Retrieved Sources ({len(msg['sources'])} chunks | Best Match: {msg.get('max_score', 0):.1%})"):
                for chunk, score in msg["sources"]:
                    st.markdown(f"**Page {chunk['page']}** *(Relevance: {score:.1%})*")
                    st.caption(chunk["text"])
                    st.markdown("---")

# Chat input
if user_question := st.chat_input("Ask anything about your uploaded PDF..."):
    if not st.session_state.indexed_pdf_name or st.session_state.chunk_embeddings is None:
        st.warning("⚠️ Please upload and index a PDF first using the sidebar!")
    elif client is None:
        st.error("⚠️ Gemini API key is missing. Please check your `.env` file.")
    else:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document & generating answer..."):
                try:
                    retrieved_chunks, scores, max_score = retrieve(
                        client=client,
                        query=user_question,
                        chunks=st.session_state.chunks,
                        chunk_embeddings=st.session_state.chunk_embeddings,
                        top_k=TOP_K,
                    )

                    # 1. Similarity Threshold Guard: "I don't know" rule
                    if max_score < SIMILARITY_THRESHOLD:
                        answer_text = (
                            "I could not find information in the provided document to answer "
                            "your question. (Relevance score was below threshold)."
                        )
                        st.markdown(answer_text)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer_text,
                            "sources": [],
                            "max_score": max_score,
                        })
                    else:
                        # 2. Build grounded prompt with explicit page badges
                        context_snippets = []
                        for i, (chunk, score) in enumerate(zip(retrieved_chunks, scores), start=1):
                            snippet = f"[Source {i} | Page {chunk.page}]\n{chunk.text}"
                            context_snippets.append(snippet)
                        context_block = "\n\n---\n\n".join(context_snippets)

                        grounded_message = (
                            f"### Document Context\n\n{context_block}\n\n"
                            f"### User Question\n{user_question}"
                        )

                        st.session_state.api_history.append(
                            types.Content(role="user", parts=[types.Part(text=grounded_message)])
                        )

                        response = client.models.generate_content(
                            model=CHAT_MODEL,
                            contents=st.session_state.api_history,
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_PROMPT,
                            ),
                        )
                        answer_text = response.text

                        st.session_state.api_history.append(
                            types.Content(role="model", parts=[types.Part(text=answer_text)])
                        )

                        # Render response
                        st.markdown(answer_text)

                        # Render sources expander
                        sources_data = [
                            ({"page": c.page, "text": c.text}, s)
                            for c, s in zip(retrieved_chunks, scores)
                        ]
                        with st.expander(f"📚 View Retrieved Sources ({len(sources_data)} chunks | Best Match: {max_score:.1%})"):
                            for chunk_info, score in sources_data:
                                st.markdown(f"**Page {chunk_info['page']}** *(Relevance: {score:.1%})*")
                                st.caption(chunk_info["text"])
                                st.markdown("---")

                        # Save to message history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer_text,
                            "sources": sources_data,
                            "max_score": max_score,
                        })

                except Exception as e:
                    err_text = f"❌ An error occurred: {e}"
                    st.error(err_text)
                    st.session_state.messages.append({"role": "assistant", "content": err_text})
