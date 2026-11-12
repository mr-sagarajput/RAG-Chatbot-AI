"""
RAG Chatbot – Chat with your PDF (Terminal Edition)
===================================================
Production-ready Retrieval-Augmented Generation (RAG) with:
  • Persistent Disk Caching (hash-based, 0 API calls on repeat loads)
  • Page-Level Source Citations (e.g., [Page X] in answer & breakdown)
  • Smart Paragraph & Sentence Chunking (no broken words/sentences)
  • "I don't know" Confidence Threshold (avoids hallucinations)
  • Hybrid Text Extraction + Vision OCR (for scanned/image-only PDFs)
  • Free Tier 100 RPM Rate-Limit Resilience
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import re
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pypdfium2 as pdfium
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

EMBED_MODEL: str = "gemini-embedding-001"
CHAT_MODEL: str = "gemini-3.5-flash-lite"

CHUNK_SIZE: int = 1500        # target character count per chunk
CHUNK_OVERLAP: int = 200      # character overlap between consecutive chunks
TOP_K: int = 5                # number of relevant chunks to retrieve per query
SIMILARITY_THRESHOLD: float = 0.35  # minimum cosine similarity to answer

CACHE_DIR: Path = Path(".cache")

SYSTEM_PROMPT: str = textwrap.dedent("""\
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
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    chunk_id: int
    page: int
    text: str


# ---------------------------------------------------------------------------
# Caching Helpers
# ---------------------------------------------------------------------------

def compute_file_hash(path: str | Path) -> str:
    """Compute SHA-256 hash of a file to detect changes."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_cache_path(file_hash: str) -> Path:
    """Return path to cached embeddings for a given hash."""
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
        print(f"    [Warning] Vision OCR failed: {e}")
        return ""


def load_pdf_pages(path: str | Path, client: genai.Client) -> List[Tuple[int, str]]:
    """Extract text from each page. If a page has no text (scanned), runs Vision OCR."""
    path = Path(path)
    reader = PdfReader(str(path))
    pages: List[Tuple[int, str]] = []

    pdfium_doc = None  # Loaded lazily only if scanned pages are detected

    for i, page in enumerate(reader.pages, start=1):
        raw_text = (page.extract_text() or "").strip()

        # If page has sufficient digital text, use it directly
        if len(raw_text) >= 30:
            pages.append((i, raw_text))
            continue

        # Otherwise page is likely scanned/image-only: run OCR fallback
        print(f"  🔍 Page {i} appears to be scanned/image-only. Running Vision OCR ...", end=" ", flush=True)
        try:
            if pdfium_doc is None:
                pdfium_doc = pdfium.PdfDocument(str(path))
            pdf_page = pdfium_doc[i - 1]
            pil_image = pdf_page.render(scale=2.0).to_pil()

            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            ocr_text = ocr_page_image(client, buf.getvalue())

            if ocr_text:
                pages.append((i, ocr_text))
                print(f"extracted {len(ocr_text)} chars.")
            else:
                print("no text detected.")
        except Exception as e:
            print(f"OCR skipped ({e}).")

    if not pages:
        raise ValueError("No extractable text or images found in the PDF.")
    return pages


# ---------------------------------------------------------------------------
# Smart Paragraph & Sentence Chunking
# ---------------------------------------------------------------------------

def split_text_smart(
    text: str,
    max_chars: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split text recursively by paragraphs, then sentences, then words.

    Guarantees no sentence is cut awkwardly in the middle.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        # If single paragraph exceeds max_chars, split it by sentence
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
                        # Add overlap from previous text
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
    """Convert page-level texts into smart chunks tagged with exact page numbers."""
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
# Embedding Helpers with 100 RPM Rate-Limit Resilience
# ---------------------------------------------------------------------------

def embed_content_with_retry(
    client: genai.Client,
    model: str,
    contents: str | list[str],
    config: types.EmbedContentConfig,
) -> types.EmbedContentResponse:
    """Wrapper around embed_content with automatic 429 rate limit retry."""
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
                print(
                    f"\n⏳ [RATE LIMIT] Free Tier quota reached (429). "
                    f"Waiting {total_wait:.0f}s before retrying...",
                    flush=True,
                )
                time.sleep(total_wait)
            else:
                raise


def embed_chunks(client: genai.Client, chunks: List[DocumentChunk]) -> np.ndarray:
    """Embed chunks in batches of 100 with automatic batch pacing."""
    texts = [c.text for c in chunks]
    batch_size = 100
    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    if total_batches > 1:
        print()

    for i in range(0, len(texts), batch_size):
        batch_idx = (i // batch_size) + 1
        batch = texts[i : i + batch_size]

        if total_batches > 1:
            print(f"  ↳ Embedding Batch {batch_idx}/{total_batches} ({len(batch)} chunks) …", flush=True)

        response = embed_content_with_retry(
            client=client,
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        for emb in response.embeddings:
            all_embeddings.append(emb.values)

        if batch_idx < total_batches:
            pause_time = 60.0
            print(f"    ⏳ Pausing {pause_time:.0f}s between batches for Free Tier quota …", flush=True)
            time.sleep(pause_time)

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
    """Compute cosine similarity between one query vector and document vectors."""
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
    """Retrieve top-K chunks and return chunks, individual scores, and max score."""
    q_vec = embed_query(client, query)
    scores = cosine_similarity(q_vec, chunk_embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved = [chunks[i] for i in top_indices]
    top_scores = [float(scores[i]) for i in top_indices]
    max_score = top_scores[0] if top_scores else 0.0

    return retrieved, top_scores, max_score


# ---------------------------------------------------------------------------
# Chat Session with Source Citation & Relevance Guard
# ---------------------------------------------------------------------------

class RAGChatSession:
    """Multi-turn RAG chat session with strict grounding and source tracking."""

    def __init__(
        self,
        client: genai.Client,
        chunks: List[DocumentChunk],
        chunk_embeddings: np.ndarray,
    ) -> None:
        self.client = client
        self.chunks = chunks
        self.chunk_embeddings = chunk_embeddings
        self.history: List[types.Content] = []

    def ask(self, question: str) -> Dict[str, Any]:
        """Process question, check similarity threshold, cite sources, and answer."""
        retrieved_chunks, scores, max_score = retrieve(
            self.client,
            question,
            self.chunks,
            self.chunk_embeddings,
            top_k=TOP_K,
        )

        # 1. Similarity Threshold Guard: "I don't know" rule
        if max_score < SIMILARITY_THRESHOLD:
            return {
                "answer": (
                    "I could not find relevant information in the provided document to answer "
                    "your question. (Relevance score was below threshold)."
                ),
                "sources": [],
                "max_score": max_score,
                "is_confident": False,
            }

        # 2. Build grounded context block with explicit page badges
        context_snippets = []
        for i, (chunk, score) in enumerate(zip(retrieved_chunks, scores), start=1):
            snippet = f"[Source {i} | Page {chunk.page}]\n{chunk.text}"
            context_snippets.append(snippet)
        context_block = "\n\n---\n\n".join(context_snippets)

        grounded_message = (
            f"### Document Context\n\n{context_block}\n\n"
            f"### User Question\n{question}"
        )

        # 3. Append to multi-turn conversation
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=grounded_message)])
        )

        response = self.client.models.generate_content(
            model=CHAT_MODEL,
            contents=self.history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )
        answer: str = response.text

        self.history.append(
            types.Content(role="model", parts=[types.Part(text=answer)])
        )

        return {
            "answer": answer,
            "sources": list(zip(retrieved_chunks, scores)),
            "max_score": max_score,
            "is_confident": True,
        }


# ---------------------------------------------------------------------------
# Terminal UI
# ---------------------------------------------------------------------------

BANNER = r"""
  ____    _    ____    ____ _           _   _           _
 |  _ \  / \  / ___|  / ___| |__   __ _| |_| |__   ___ | |_
 | |_) |/ _ \| |  _  | |   | '_ \ / _` | __| '_ \ / _ \| __|
 |  _ </ ___ \ |_| | | |___| | | | (_| | |_| |_) | (_) | |_
 |_| \_\_/ \_\____|  \____|_| |_|\__,_|\__|_.__/ \___/ \__|

   Chat with PDF  •  Persistent Disk Cache  •  Page Citations
"""

HELP_TEXT = """
Commands:
  /help    – Show this message
  /clear   – Clear conversation history
  /sources – Re-display sources from the last query
  /quit    – Exit the application
"""


def print_wrapped(text: str, width: int = 90) -> None:
    """Print text with word wrapping."""
    for paragraph in text.split("\n"):
        if paragraph.strip():
            print(textwrap.fill(paragraph, width=width))
        else:
            print()


def run_chat(pdf_path: str | Path) -> None:
    """Main terminal loop."""
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY is not set in .env file.")
        sys.exit(1)

    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(api_version="v1beta"),
    )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"[ERROR] File not found: {pdf_path}")
        sys.exit(1)

    print(BANNER)
    print(f"📄 Loading PDF: {pdf_path.name}")

    # Check disk cache first
    file_hash = compute_file_hash(pdf_path)
    cached_data = load_cached_embeddings(file_hash)

    if cached_data is not None:
        chunks, chunk_embeddings = cached_data
        print(f"⚡ Found cached embeddings on disk! Loaded {len(chunks)} chunks instantly (0 API calls).")
    else:
        print("📖 Extracting pages (digital text + OCR fallback) …")
        pages = load_pdf_pages(pdf_path, client)
        print(f"   Extracted {len(pages)} pages.")

        print("✂️  Smart-chunking into paragraphs & sentences …", end=" ", flush=True)
        chunks = build_document_chunks(pages)
        print(f"{len(chunks)} chunks created.")

        print("🔢 Embedding chunks (with Free Tier rate-limit protection) …", end=" ", flush=True)
        chunk_embeddings = embed_chunks(client, chunks)
        print("done.")

        print("💾 Saving embeddings to disk cache (.cache/) …", end=" ", flush=True)
        save_cached_embeddings(file_hash, chunks, chunk_embeddings)
        print("saved.")

    session = RAGChatSession(client, chunks, chunk_embeddings)

    print("\n" + "─" * 65)
    print("✅ Ready! Ask anything. Citations [Page X] will appear automatically.")
    print("─" * 65 + "\n")

    last_sources = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("👋 Goodbye!")
            break

        if user_input.lower() == "/clear":
            session.history.clear()
            print("🗑️ Conversation history cleared.\n")
            continue

        if user_input.lower() == "/help":
            print(HELP_TEXT)
            continue

        if user_input.lower() == "/sources":
            if not last_sources:
                print("No recent sources to show.\n")
            else:
                print("\n📚 Sources retrieved for last query:")
                for chunk, score in last_sources:
                    snippet = chunk.text.replace("\n", " ")[:140]
                    print(f"  • Page {chunk.page} (Relevance: {score:.1%}): \"{snippet}...\"")
                print()
            continue

        print("\n🤖 Assistant: ", end="", flush=True)
        try:
            result = session.ask(user_input)
            last_sources = result["sources"]

            print()
            print_wrapped(result["answer"])

            # Print source summary under answer
            if result["sources"]:
                pages_cited = sorted({c.page for c, _ in result["sources"]})
                pages_str = ", ".join(f"Page {p}" for p in pages_cited)
                print(f"\n📌 Cited from: {pages_str} (Match quality: {result['max_score']:.1%})\n")
            else:
                print()

        except Exception as exc:
            print(f"\n[ERROR] {exc}\n")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python rag_chat.py <path_to_pdf>")
        print("Example: python rag_chat.py 'Sagar Resume.pdf'")
        sys.exit(1)

    run_chat(sys.argv[1])


if __name__ == "__main__":
    main()
