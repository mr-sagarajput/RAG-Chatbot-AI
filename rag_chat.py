"""
RAG Chatbot – Chat with your PDF
=================================
Uses:
  • pypdf              – PDF text extraction
  • google-genai       – Gemini embeddings (gemini-embedding-001) + generative
                        model (gemini-3.5-flash-lite)
  • numpy              – cosine-similarity search over chunk embeddings

Flow:
  1. Load & chunk the PDF.
  2. Embed every chunk with gemini-embedding-001.
  3. On each user query, embed the query, find the top-k most similar
     chunks, build a grounded prompt, and get the answer via gemini-3.5-flash-lite.
"""

from __future__ import annotations

import os
import re
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import errors, types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Embedding model: gemini-embedding-001 is the supported Google GenAI embedding model
EMBED_MODEL: str = "gemini-embedding-001"

# Chat model: gemini-3.5-flash-lite as requested
CHAT_MODEL: str = "gemini-3.5-flash-lite"

CHUNK_SIZE: int = 2000       # characters per chunk
CHUNK_OVERLAP: int = 250     # character overlap between consecutive chunks
TOP_K: int = 5               # number of chunks to retrieve per query

SYSTEM_PROMPT: str = textwrap.dedent("""\
    You are a helpful assistant that answers questions strictly based on the
    provided document context. If the answer is not found in the context,
    say so clearly. Do not make up information.
    Always cite relevant details from the context to support your answer.
""")


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def load_pdf(path: str | Path) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i}]\n{text}")
    if not pages:
        raise ValueError("No extractable text found in the PDF.")
    return "\n\n".join(pages)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping character-level chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Embedding helpers  (google-genai SDK)
# ---------------------------------------------------------------------------

def embed_content_with_retry(
    client: genai.Client,
    model: str,
    contents: str | list[str],
    config: types.EmbedContentConfig,
) -> types.EmbedContentResponse:
    """Wrapper around client.models.embed_content with automatic 429/RESOURCE_EXHAUSTED retry.

    Handles Gemini Free Tier rate limits (100 requests/minute). Parses the
    suggested retry delay from ClientError/APIError or defaults to 60 seconds,
    adds a 2-second buffer, and automatically retries.
    """
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
                # Default wait is 60 seconds if not specified in the error message
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

                total_wait = wait_sec + 2.0  # +2s safety buffer
                print(
                    f"\n⏳ [RATE LIMIT] Free Tier rate limit reached (429 RESOURCE_EXHAUSTED). "
                    f"Waiting {total_wait:.1f}s for quota to reset before retrying...",
                    flush=True,
                )
                time.sleep(total_wait)
            else:
                raise


def embed_texts(client: genai.Client, texts: list[str]) -> np.ndarray:
    """Return a (N, D) float32 array of embeddings for a list of texts.

    Uses task_type RETRIEVAL_DOCUMENT so the model optimises for
    passage/document retrieval. Batches in groups of up to 100 chunks.
    If multiple batches are required, prints batch progress and automatically
    pauses between batches to respect the Free Tier 100 RPM limit.
    """
    batch_size = 100
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    if total_batches > 1:
        print()  # Move to a new line for batch reporting

    for i in range(0, len(texts), batch_size):
        batch_idx = (i // batch_size) + 1
        batch = texts[i : i + batch_size]

        if total_batches > 1:
            print(f"  ↳ Batch {batch_idx}/{total_batches} ({len(batch)} chunks) …", flush=True)

        response = embed_content_with_retry(
            client=client,
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        for emb in response.embeddings:
            all_embeddings.append(emb.values)

        # Pause automatically between batches if there are more batches to respect the 100 RPM quota
        if batch_idx < total_batches:
            pause_time = 60.0
            print(f"    ⏳ Pausing {pause_time:.0f}s between batches to respect Free Tier 100 RPM rate limit …", flush=True)
            time.sleep(pause_time)

    return np.array(all_embeddings, dtype=np.float32)


def embed_query(client: genai.Client, query: str) -> np.ndarray:
    """Return a 1-D float32 embedding for a single query string.

    Uses task_type RETRIEVAL_QUERY so cosine similarity with
    RETRIEVAL_DOCUMENT vectors is well-calibrated.
    """
    response = embed_content_with_retry(
        client=client,
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return np.array(response.embeddings[0].values, dtype=np.float32)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one query vector and many doc vectors."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return d @ q


def retrieve(
    client: genai.Client,
    query: str,
    chunks: list[str],
    chunk_embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> list[str]:
    """Return the *top_k* most relevant chunks for the query."""
    q_vec = embed_query(client, query)
    scores = cosine_similarity(q_vec, chunk_embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Chat session
# ---------------------------------------------------------------------------

class RAGChatSession:
    """Maintains multi-turn conversation history with RAG context injection."""

    def __init__(
        self,
        client: genai.Client,
        chunks: list[str],
        chunk_embeddings: np.ndarray,
    ) -> None:
        self.client = client
        self.chunks = chunks
        self.chunk_embeddings = chunk_embeddings
        # history: list of Content objects for multi-turn
        self.history: list[types.Content] = []

    def ask(self, question: str) -> str:
        """Retrieve context, build a grounded message, and return the answer."""
        # 1. Retrieve relevant chunks
        relevant = retrieve(self.client, question, self.chunks, self.chunk_embeddings)
        context_block = "\n\n---\n\n".join(relevant)

        # 2. Build the user turn with injected context
        grounded_message = (
            f"### Document Context\n\n{context_block}\n\n"
            f"### Question\n\n{question}"
        )

        # 3. Append user turn to history
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=grounded_message)])
        )

        # 4. Send to the model (keeps full conversation history)
        response = self.client.models.generate_content(
            model=CHAT_MODEL,
            contents=self.history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )
        answer: str = response.text

        # 5. Store assistant reply in history
        self.history.append(
            types.Content(role="model", parts=[types.Part(text=answer)])
        )
        return answer


# ---------------------------------------------------------------------------
# Terminal UI
# ---------------------------------------------------------------------------

BANNER = r"""
  ____    _    ____    ____ _           _   _           _
 |  _ \  / \  / ___|  / ___| |__   __ _| |_| |__   ___ | |_
 | |_) |/ _ \| |  _  | |   | '_ \ / _` | __| '_ \ / _ \| __|
 |  _ </ ___ \ |_| | | |___| | | | (_| | |_| |_) | (_) | |_
 |_| \_\_/ \_\____|  \____|_| |_|\__,_|\__|_.__/ \___/ \__|

           Chat with your PDF  •  Powered by Gemini 3.5 Flash-Lite
"""

HELP_TEXT = """
Commands:
  /help    – show this message
  /clear   – clear conversation history
  /quit    – exit the application
"""


def print_wrapped(text: str, width: int = 90) -> None:
    """Print text with soft word-wrapping."""
    for paragraph in text.split("\n"):
        if paragraph.strip():
            print(textwrap.fill(paragraph, width=width))
        else:
            print()


def run_chat(pdf_path: str | Path) -> None:
    """Main entry point: load PDF, build index, run chat loop."""

    # --- API key check -------------------------------------------------
    if not GEMINI_API_KEY:
        print(
            "[ERROR] GEMINI_API_KEY is not set.\n"
            "        Create a .env file with GEMINI_API_KEY=<your_key>\n"
            "        or export it as an environment variable."
        )
        sys.exit(1)

    # --- Initialise the new google-genai client -----------------------
    # Use v1beta endpoint to ensure latest model versions and methods are available
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(api_version="v1beta"),
    )

    # --- Load & index --------------------------------------------------
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"[ERROR] File not found: {pdf_path}")
        sys.exit(1)

    print(BANNER)
    print(f"📄  Loading PDF: {pdf_path.name}")
    full_text = load_pdf(pdf_path)

    print(f"✂️   Chunking text …", end=" ", flush=True)
    chunks = chunk_text(full_text)
    print(f"{len(chunks)} chunks created.")

    print("🔢  Embedding chunks (this may take a moment for large PDFs) …", end=" ", flush=True)
    chunk_embeddings = embed_texts(client, chunks)
    print("done.")

    session = RAGChatSession(client, chunks, chunk_embeddings)

    print("\n" + "─" * 60)
    print("✅  Ready! Ask anything about your PDF. Type /help for commands.")
    print("─" * 60 + "\n")

    # --- Chat loop -----------------------------------------------------
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋  Goodbye!")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("👋  Goodbye!")
            break

        if user_input.lower() == "/clear":
            session.history.clear()
            print("🗑️   Conversation history cleared.\n")
            continue

        if user_input.lower() == "/help":
            print(HELP_TEXT)
            continue

        # Regular question
        print("\n🤖  Assistant: ", end="", flush=True)
        try:
            answer = session.ask(user_input)
            print()
            print_wrapped(answer)
            print()
        except Exception as exc:  # noqa: BLE001
            print(f"\n[ERROR] {exc}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python rag_chat.py <path_to_pdf>")
        print("Example: python rag_chat.py document.pdf")
        sys.exit(1)

    run_chat(sys.argv[1])


if __name__ == "__main__":
    main()