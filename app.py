"""
RAG Chatbot – Web Interface (Streamlit)
========================================
Interactive ChatGPT-like web UI for chatting with your PDF documents.
Powered by Google Gemini:
  • Embedding Model: gemini-embedding-001
  • Chat Model:      gemini-3.5-flash-lite
"""

from __future__ import annotations

import io
import os
import re
import textwrap
import time
from typing import List

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import errors, types

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
# Constants
# ---------------------------------------------------------------------------

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 250
TOP_K = 5

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a helpful assistant that answers questions strictly based on the
    provided document context. If the answer is not found in the context,
    say so clearly. Do not make up information.
    Always cite relevant details from the context to support your answer.
""")


# ---------------------------------------------------------------------------
# PDF & RAG Helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_file) -> str:
    """Extract page-tagged text from an uploaded PDF file."""
    reader = PdfReader(pdf_file)
    pages: List[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i}]\n{text}")
    if not pages:
        raise ValueError("No extractable text found in the uploaded PDF.")
    return "\n\n".join(pages)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character-level chunks."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += size - overlap
    return chunks


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
                msg = f"⏳ Free Tier quota reached (429). Waiting {total_wait:.0f}s before retrying..."
                if status_placeholder:
                    status_placeholder.warning(msg)
                time.sleep(total_wait)
            else:
                raise


def embed_texts(client: genai.Client, texts: List[str], progress_callback=None) -> np.ndarray:
    """Embed chunks in batches of up to 100 with Free Tier rate-limit protection."""
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

        # Pause 60s between batches if there are more batches
        if batch_idx < total_batches:
            if progress_callback:
                progress_callback(batch_idx, total_batches, len(batch), is_pausing=True)
            time.sleep(60.0)

    return np.array(all_embeddings, dtype=np.float32)


def embed_query(client: genai.Client, query: str) -> np.ndarray:
    """Embed a single search query."""
    response = embed_content_with_retry(
        client=client,
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return np.array(response.embeddings[0].values, dtype=np.float32)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all document vectors."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return d @ q


def retrieve(
    client: genai.Client,
    query: str,
    chunks: List[str],
    chunk_embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> List[str]:
    """Retrieve top-K most similar text chunks for the query."""
    q_vec = embed_query(client, query)
    scores = cosine_similarity(q_vec, chunk_embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_history" not in st.session_state:
    st.session_state.api_history = []

if "indexed_pdf_name" not in st.session_state:
    st.session_state.indexed_pdf_name = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = None

# Initialize Gemini client
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
    st.caption("Powered by Google Gemini")
    st.markdown("---")

    # API Key check
    if not GEMINI_API_KEY:
        st.error("⚠️ `GEMINI_API_KEY` not found in `.env` file.")
        user_key = st.text_input("Enter Gemini API Key:", type="password")
        if user_key:
            client = get_gemini_client(user_key)
            GEMINI_API_KEY = user_key
    else:
        st.success("🔑 API Key loaded from `.env`", icon="✅")

    st.markdown("### 📥 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None and client is not None:
        # Check if this PDF is already indexed
        if st.session_state.indexed_pdf_name != uploaded_file.name:
            status_box = st.status(f"Processing '{uploaded_file.name}'...", expanded=True)
            try:
                status_box.write("📖 Extracting text from PDF...")
                pdf_text = extract_text_from_pdf(uploaded_file)

                status_box.write("✂️ Splitting into chunks...")
                chunks = chunk_text(pdf_text)
                status_box.write(f"Created {len(chunks)} text chunks.")

                def update_progress(batch_num, total_batches, batch_len, is_pausing=False):
                    if is_pausing:
                        status_box.write(f"⏳ Free Tier limit: Pausing 60s before Batch {batch_num + 1}...")
                    else:
                        status_box.write(f"🔢 Embedding Batch {batch_num}/{total_batches} ({batch_len} chunks)...")

                chunk_embeddings = embed_texts(client, chunks, progress_callback=update_progress)

                # Store in session state
                st.session_state.chunks = chunks
                st.session_state.chunk_embeddings = chunk_embeddings
                st.session_state.indexed_pdf_name = uploaded_file.name
                st.session_state.messages = []  # Reset chat on new file
                st.session_state.api_history = []

                status_box.update(label="✅ PDF Indexed Successfully!", state="complete", expanded=False)
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

# If no PDF is uploaded yet, show welcome instructions
if not st.session_state.indexed_pdf_name:
    st.info("👈 **To get started:** Upload a PDF file from the sidebar on the left.")
    st.markdown(
        """
        ### What this app does:
        1. 📄 **Reads your PDF** (even multi-page / large documents)
        2. 🔍 **Finds the most relevant passages** using `gemini-embedding-001`
        3. 🤖 **Answers your questions accurately** with `gemini-3.5-flash-lite`, grounded in your PDF's exact text!
        """
    )

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_question := st.chat_input("Ask anything about your uploaded PDF..."):
    if not st.session_state.indexed_pdf_name or st.session_state.chunk_embeddings is None:
        st.warning("⚠️ Please upload and index a PDF first using the sidebar!")
    elif client is None:
        st.error("⚠️ Gemini API key is missing. Please check your `.env` file.")
    else:
        # 1. Display user message immediately
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # 2. Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document & generating answer..."):
                try:
                    # Retrieve top chunks
                    relevant_chunks = retrieve(
                        client=client,
                        query=user_question,
                        chunks=st.session_state.chunks,
                        chunk_embeddings=st.session_state.chunk_embeddings,
                        top_k=TOP_K,
                    )
                    context_block = "\n\n---\n\n".join(relevant_chunks)

                    # Build grounded prompt
                    grounded_message = (
                        f"### Document Context\n\n{context_block}\n\n"
                        f"### Question\n\n{user_question}"
                    )

                    # Append to API turn history
                    st.session_state.api_history.append(
                        types.Content(role="user", parts=[types.Part(text=grounded_message)])
                    )

                    # Call Gemini 3.5 Flash-Lite
                    response = client.models.generate_content(
                        model=CHAT_MODEL,
                        contents=st.session_state.api_history,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                        ),
                    )
                    answer_text = response.text

                    # Save assistant turn in API history
                    st.session_state.api_history.append(
                        types.Content(role="model", parts=[types.Part(text=answer_text)])
                    )

                    # Render answer
                    st.markdown(answer_text)

                    # Save in UI message history
                    st.session_state.messages.append({"role": "assistant", "content": answer_text})

                except Exception as e:
                    err_text = f"❌ An error occurred: {e}"
                    st.error(err_text)
                    st.session_state.messages.append({"role": "assistant", "content": err_text})
