# ============================================================
# AI ASSISTANT BACKEND
# FastAPI + DeepSeek + SQLite + Chroma RAG + Memory
# ============================================================

from __future__ import annotations

import os
import re
import io
import json
import base64
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from typing import Optional, Any, Generator

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ENV_FILE = BASE_DIR / ".env"

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "chatbot.db"

load_dotenv(ENV_FILE)


# ============================================================
# HUGGING FACE CONFIGURATION
# ============================================================

HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

if not HF_TOKEN:
    raise ValueError(
        f"\nHF_TOKEN is missing.\n\n"
        f"Create this file:\n{ENV_FILE}\n\n"
        "Add:\n"
        "HF_TOKEN=your_new_huggingface_token\n"
    )


MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
)

MAX_TOKENS = int(
    os.getenv("MAX_TOKENS", "1024")
)

TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0.3")
)


# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    api_key=HF_TOKEN
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Rules:

- Give clear and simple answers.
- Explain technical topics step by step.
- Use examples when useful.
- Keep answers relevant to the user's question.
- If you do not know something, say so.
- Do not invent facts.
- When an image is provided, analyze it carefully.
- When document context is provided, use it when relevant.
- If the answer comes from uploaded documents, mention the
  relevant document name when useful.
- Do not claim that you searched the web unless web-search
  context was actually provided.
- Prefer practical and accurate answers.
"""


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Assistant API",
    description="DeepSeek AI Assistant with RAG and Memory",
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL RAG STATE
# ============================================================

_chroma_client = None
_rag_collection = None
_rag_error = None


# ============================================================
# SQLITE
# ============================================================

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
    )

    conn.row_factory = sqlite3.Row

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    return conn


def now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def init_database() -> None:

    conn = get_db()

    try:

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,

                FOREIGN KEY(chat_id)
                REFERENCES chats(id)
                ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                file_path TEXT NOT NULL,
                chunks INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_chat
            ON messages(chat_id);

            CREATE INDEX IF NOT EXISTS idx_memories_created
            ON memories(created_at);

            CREATE INDEX IF NOT EXISTS idx_chats_updated
            ON chats(updated_at);

            CREATE INDEX IF NOT EXISTS idx_complaints_created
            ON complaints(created_at);
            """
        )

        conn.commit()

    finally:
        conn.close()


init_database()


# ============================================================
# CHROMA RAG INITIALIZATION
# ============================================================

def get_rag_collection():
    """
    Initialize Chroma only once.

    Returns:
        Chroma collection or None if initialization fails.
    """

    global _chroma_client
    global _rag_collection
    global _rag_error

    if _rag_collection is not None:
        return _rag_collection

    try:

        print()
        print("📚 Initializing Chroma RAG...")
        print(
            f"   Chroma DB : {CHROMA_DIR}"
        )
        print(
            f"   Embedding : {EMBEDDING_MODEL}"
        )

        import chromadb

        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction
        )

        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        embedding_function = (
            SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )
        )

        _rag_collection = (
            _chroma_client.get_or_create_collection(
                name="documents",
                embedding_function=embedding_function,
            )
        )

        _rag_error = None

        print("✅ Chroma RAG initialized successfully.")
        print(
            f"   Documents indexed: "
            f"{_rag_collection.count()}"
        )
        print()

        return _rag_collection

    except Exception as error:

        _rag_error = str(error)

        _rag_collection = None

        print()
        print("❌ Chroma RAG initialization failed.")
        print(f"   Error: {error}")
        print()

        return None


# ============================================================
# FASTAPI STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    print()
    print("=" * 70)
    print("🚀 AI ASSISTANT STARTUP")
    print("=" * 70)

    print(
        f"Model    : {MODEL_NAME}"
    )

    print(
        f"Database : {DB_PATH}"
    )

    print(
        f"Chroma   : {CHROMA_DIR}"
    )

    print()

    # ============================================================
    # IMPORTANT:
    # Do NOT initialize RAG at startup.
    #
    # get_rag_collection() loads the local embedding model
    # (all-MiniLM-L6-v2), which can consume a lot of RAM on
    # Render's 512 MB instance.
    #
    # RAG will be initialized only when it is actually needed.
    # ============================================================

    print("ℹ️ RAG initialization: Lazy loading enabled")

    print("=" * 70)
    print("✅ Startup completed")
    print("=" * 70)
    print()

# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[str]:

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:
        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(
    file_bytes: bytes
) -> str:

    try:

        from pypdf import PdfReader

        reader = PdfReader(
            io.BytesIO(file_bytes)
        )

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                text = page.extract_text()

                if text:

                    pages.append(
                        f"\n[Page {page_number}]\n{text}"
                    )

            except Exception as error:

                print(
                    f"⚠️ Could not extract page "
                    f"{page_number}: {error}"
                )

        return "\n".join(
            pages
        ).strip()

    except Exception as error:

        raise RuntimeError(
            f"PDF extraction failed: {error}"
        )


# ============================================================
# DOCUMENT SEARCH
# ============================================================

def search_documents(
    query: str,
    n_results: int = 5,
) -> list[dict]:

    collection = get_rag_collection()

    if collection is None:
        return []

    try:

        count = collection.count()

        if count == 0:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(
                n_results,
                count
            ),
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        output = []

        for index, document in enumerate(
            documents
        ):

            metadata = {}

            if index < len(metadatas):

                metadata = (
                    metadatas[index]
                    or {}
                )

            output.append(
                {
                    "text": document,
                    "filename": metadata.get(
                        "filename",
                        "Unknown document"
                    ),
                    "chunk": metadata.get(
                        "chunk_index",
                        0
                    ),
                }
            )

        return output

    except Exception as error:

        print(
            f"⚠️ RAG search error: {error}"
        )

        return []


# ============================================================
# MEMORY SYSTEM
# ============================================================

MEMORY_PATTERNS = [

    (
        r"\bmy name is\b",
        "user_fact"
    ),

    (
        r"\bi am\b",
        "user_fact"
    ),

    (
        r"\bi'm\b",
        "user_fact"
    ),

    (
        r"\bmy favorite\b",
        "preference"
    ),

    (
        r"\bi like\b",
        "preference"
    ),

    (
        r"\bi love\b",
        "preference"
    ),

    (
        r"\bi prefer\b",
        "preference"
    ),

    (
        r"\bi study\b",
        "education"
    ),

    (
        r"\bi work\b",
        "work"
    ),

    (
        r"\bmy goal\b",
        "goal"
    ),

    (
        r"\bremember that\b",
        "explicit_memory"
    ),

    (
        r"\bremember\b",
        "explicit_memory"
    ),
]


def extract_memory(
    text: str
) -> Optional[tuple[str, str]]:

    clean = text.strip()

    if not clean:
        return None

    if len(clean) > 500:
        return None

    lower = clean.lower()

    for pattern, memory_type in MEMORY_PATTERNS:

        if re.search(
            pattern,
            lower
        ):

            return (
                memory_type,
                clean
            )

    return None


def save_memory(
    memory_type: str,
    content: str
):

    conn = get_db()

    try:

        existing = conn.execute(
            """
            SELECT id
            FROM memories
            WHERE content = ?
            LIMIT 1
            """,
            (content,)
        ).fetchone()

        if existing:
            return

        conn.execute(
            """
            INSERT INTO memories
            (
                memory_type,
                content,
                created_at
            )
            VALUES (?, ?, ?)
            """,
            (
                memory_type,
                content,
                now_iso(),
            )
        )

        conn.commit()

    finally:
        conn.close()


def search_memories(
    query: str = "",
    limit: int = 8,
):

    conn = get_db()

    try:

        if not query.strip():

            rows = conn.execute(
                """
                SELECT *
                FROM memories
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()

        else:

            words = [
                word
                for word in re.findall(
                    r"\w+",
                    query.lower()
                )
                if len(word) > 2
            ]

            if not words:

                rows = conn.execute(
                    """
                    SELECT *
                    FROM memories
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,)
                ).fetchall()

            else:

                conditions = []
                values = []

                for word in words[:8]:

                    conditions.append(
                        "LOWER(content) LIKE ?"
                    )

                    values.append(
                        f"%{word}%"
                    )

                sql = f"""
                    SELECT *
                    FROM memories
                    WHERE {" OR ".join(conditions)}
                    ORDER BY id DESC
                    LIMIT ?
                """

                values.append(limit)

                rows = conn.execute(
                    sql,
                    values
                ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        conn.close()


# ============================================================
# CHAT DATABASE
# ============================================================

def create_chat(
    title: str = "New Chat"
):

    chat_id = str(uuid4())

    timestamp = now_iso()

    conn = get_db()

    try:

        conn.execute(
            """
            INSERT INTO chats
            (
                id,
                title,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                chat_id,
                title[:100],
                timestamp,
                timestamp,
            )
        )

        conn.commit()

    finally:
        conn.close()

    return {
        "id": chat_id,
        "title": title[:100],
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def get_chat(
    chat_id: str
):

    conn = get_db()

    try:

        row = conn.execute(
            """
            SELECT *
            FROM chats
            WHERE id = ?
            """,
            (chat_id,)
        ).fetchone()

        if not row:
            return None

        return dict(row)

    finally:
        conn.close()


def update_chat_title(
    chat_id: str,
    title: str
):

    conn = get_db()

    try:

        conn.execute(
            """
            UPDATE chats
            SET title = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                title[:100],
                now_iso(),
                chat_id,
            )
        )

        conn.commit()

    finally:
        conn.close()


def save_message(
    chat_id: str,
    role: str,
    content: str,
):

    conn = get_db()

    try:

        cursor = conn.execute(
            """
            INSERT INTO messages
            (
                chat_id,
                role,
                content,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                chat_id,
                role,
                content,
                now_iso(),
            )
        )

        conn.execute(
            """
            UPDATE chats
            SET updated_at = ?
            WHERE id = ?
            """,
            (
                now_iso(),
                chat_id,
            )
        )

        conn.commit()

        return cursor.lastrowid

    finally:
        conn.close()


def get_messages(
    chat_id: str,
    limit: int = 30,
    before_id: Optional[int] = None,
):

    conn = get_db()

    try:

        if before_id is None:

            rows = conn.execute(
                """
                SELECT *
                FROM messages
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (
                    chat_id,
                    limit,
                )
            ).fetchall()

        else:

            rows = conn.execute(
                """
                SELECT *
                FROM messages
                WHERE chat_id = ?
                AND id < ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (
                    chat_id,
                    before_id,
                    limit,
                )
            ).fetchall()

        rows = list(
            reversed(rows)
        )

        return [
            dict(row)
            for row in rows
        ]

    finally:
        conn.close()


def delete_message(
    message_id: int
):

    conn = get_db()

    try:

        conn.execute(
            """
            DELETE FROM messages
            WHERE id = ?
            """,
            (message_id,)
        )

        conn.commit()

    finally:
        conn.close()


# ============================================================
# TITLE GENERATOR
# ============================================================

def generate_title(
    text: str
) -> str:

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:
        return "New Chat"

    if len(text) <= 45:
        return text

    return text[:45].rstrip() + "..."


# ============================================================
# CLEAN MODEL RESPONSE
# ============================================================

def clean_response(
    text: str
) -> str:

    if not text:
        return ""

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(
        r"<think>.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(
        r"</?think>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


# ============================================================
# MODEL MESSAGE BUILDER
# ============================================================

def build_model_messages(
    chat_id: str,
    user_query: str,
    current_content: Any = None,
    exclude_message_id: Optional[int] = None,
):

    model_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]


    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    memories = search_memories(
        user_query,
        limit=6
    )

    if memories:

        memory_text = "\n".join(
            [
                f"- {item['content']}"
                for item in memories
            ]
        )

        model_messages.append(
            {
                "role": "system",
                "content": (
                    "Relevant user memory:\n"
                    f"{memory_text}"
                ),
            }
        )


    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    rag_results = search_documents(
        user_query,
        n_results=5
    )

    if rag_results:

        context_parts = []

        for item in rag_results:

            context_parts.append(
                "\n".join(
                    [
                        f"Source: {item['filename']}",
                        f"Chunk: {item['chunk']}",
                        item["text"],
                    ]
                )
            )

        rag_context = "\n\n".join(
            context_parts
        )

        model_messages.append(
            {
                "role": "system",
                "content": (
                    "The following information comes "
                    "from uploaded documents.\n\n"
                    "Use it when it is relevant to "
                    "the user's question.\n\n"
                    f"{rag_context}"
                ),
            }
        )


    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    history = get_messages(
        chat_id,
        limit=20,
        before_id=exclude_message_id,
    )

    for message in history:

        role = message["role"]

        content = message["content"]

        if role not in {
            "user",
            "assistant",
        }:
            continue

        model_messages.append(
            {
                "role": role,
                "content": content,
            }
        )


    # --------------------------------------------------------
    # CURRENT MESSAGE
    # --------------------------------------------------------

    if current_content is not None:

        model_messages.append(
            {
                "role": "user",
                "content": current_content,
            }
        )

    return model_messages


# ============================================================
# HF RESPONSE EXTRACTION
# ============================================================

def extract_response_text(
    response: Any
) -> str:

    try:

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if isinstance(
            content,
            str
        ):
            return content

        if isinstance(
            content,
            list
        ):

            parts = []

            for item in content:

                if isinstance(
                    item,
                    dict
                ):

                    if item.get("type") == "text":

                        parts.append(
                            item.get(
                                "text",
                                ""
                            )
                        )

            return "".join(parts)

        return str(
            content or ""
        )

    except Exception:

        return ""


def extract_stream_delta(
    chunk: Any
) -> str:

    try:

        delta = (
            chunk
            .choices[0]
            .delta
        )

        content = getattr(
            delta,
            "content",
            None
        )

        if isinstance(
            content,
            str
        ):
            return content

        if isinstance(
            content,
            list
        ):

            parts = []

            for item in content:

                if isinstance(
                    item,
                    dict
                ):

                    text = item.get(
                        "text",
                        ""
                    )

                    if text:
                        parts.append(
                            text
                        )

            return "".join(parts)

        return ""

    except Exception:

        return ""


# ============================================================
# FALLBACK STREAMING
# ============================================================

def split_for_streaming(
    text: str,
    size: int = 40
):

    for index in range(
        0,
        len(text),
        size
    ):

        yield text[
            index:index + size
        ]


# ============================================================
# MODEL GENERATOR
# ============================================================

def generate_model_stream(
    model_messages,
) -> Generator[str, None, None]:

    streamed_anything = False

    # --------------------------------------------------------
    # REAL STREAMING
    # --------------------------------------------------------

    try:

        response_stream = (
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=model_messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True,
            )
        )

        for chunk in response_stream:

            text = extract_stream_delta(
                chunk
            )

            if text:

                streamed_anything = True

                yield text

        if streamed_anything:
            return

    except Exception as error:

        if streamed_anything:
            raise error

        print(
            "⚠️ Streaming unavailable."
        )

        print(
            f"   Reason: {error}"
        )


    # --------------------------------------------------------
    # NORMAL RESPONSE FALLBACK
    # --------------------------------------------------------

    response = (
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=model_messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    )

    answer = extract_response_text(
        response
    )

    if not answer:

        raise RuntimeError(
            "Model returned an empty response."
        )

    for piece in split_for_streaming(
        answer
    ):

        yield piece


# ============================================================
# NDJSON
# ============================================================

def ndjson(
    payload: dict
) -> str:

    return (
        json.dumps(
            payload,
            ensure_ascii=False
        )
        + "\n"
    )


# ============================================================
# CHAT STREAM
# ============================================================

def stream_chat(
    chat_id: str,
    model_messages,
):

    full_answer = ""

    try:

        yield ndjson(
            {
                "type": "status",
                "message": "Generating response..."
            }
        )


        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        for chunk in generate_model_stream(
            model_messages
        ):

            full_answer += chunk

            yield ndjson(
                {
                    "type": "chunk",
                    "content": chunk,
                }
            )


        # ----------------------------------------------------
        # CLEAN
        # ----------------------------------------------------

        final_answer = clean_response(
            full_answer
        )

        if not final_answer:

            final_answer = (
                "Sorry, I could not generate "
                "a response."
            )


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_message(
            chat_id,
            "assistant",
            final_answer,
        )


        chat = get_chat(
            chat_id
        )


        # ----------------------------------------------------
        # DONE
        # ----------------------------------------------------

        yield ndjson(
            {
                "type": "done",
                "response": final_answer,
                "chat": chat,
            }
        )

    except GeneratorExit:

        return

    except Exception as error:

        print(
            f"❌ Generation error: {error}"
        )

        yield ndjson(
            {
                "type": "error",
                "message": str(error),
            }
        )


# ============================================================
# REQUEST MODELS
# ============================================================

class CreateChatRequest(BaseModel):

    title: str = "New Chat"


class MessageRequest(BaseModel):

    message: str


class RenameChatRequest(BaseModel):

    title: str


class CreateComplaintRequest(BaseModel):

    description: str
    chat_id: Optional[str] = None


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "service": "AI Assistant",
        "model": MODEL_NAME,
        "status": "running",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    # Force RAG initialization so health is accurate
    collection = get_rag_collection()

    rag_available = (
        collection is not None
    )

    rag_documents = 0

    if collection is not None:

        try:

            rag_documents = (
                collection.count()
            )

        except Exception:
            rag_documents = 0

    return {
        "status": "ok",
        "service": "AI Assistant",
        "model": MODEL_NAME,
        "database": DB_PATH.exists(),
        "rag": rag_available,
        "rag_documents": rag_documents,
        "rag_error": _rag_error,
        "embedding_model": EMBEDDING_MODEL,
        "memory": True,
    }


# ============================================================
# CREATE CHAT
# ============================================================

@app.post("/chats")
def create_new_chat(
    request: CreateChatRequest
):

    chat = create_chat(
        request.title
        or "New Chat"
    )

    return {
        "success": True,
        "chat": chat,
    }


# ============================================================
# LIST CHATS
# ============================================================

@app.get("/chats")
def list_chats():

    conn = get_db()

    try:

        rows = conn.execute(
            """
            SELECT *
            FROM chats
            ORDER BY updated_at DESC
            """
        ).fetchall()

        chats = [
            dict(row)
            for row in rows
        ]

        return {
            "success": True,
            "chats": chats,
        }

    finally:
        conn.close()


# ============================================================
# GET CHAT
# ============================================================

@app.get("/chats/{chat_id}")
def get_single_chat(
    chat_id: str
):

    chat = get_chat(
        chat_id
    )

    if not chat:

        raise HTTPException(
            status_code=404,
            detail="Chat not found."
        )

    messages = get_messages(
        chat_id,
        limit=1000
    )

    return {
        "success": True,
        "chat": chat,
        "messages": messages,
    }


# ============================================================
# SEND MESSAGE
# ============================================================

@app.post("/chats/{chat_id}/messages")
def send_message(
    chat_id: str,
    request: MessageRequest,
):

    chat = get_chat(
        chat_id
    )

    if not chat:

        raise HTTPException(
            status_code=404,
            detail="Chat not found."
        )


    user_message = (
        request.message
        .strip()
    )

    if not user_message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )


    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    message_id = save_message(
        chat_id,
        "user",
        user_message,
    )


    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    memory = extract_memory(
        user_message
    )

    if memory:

        memory_type, content = memory

        save_memory(
            memory_type,
            content
        )


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    if chat["title"] == "New Chat":

        update_chat_title(
            chat_id,
            generate_title(
                user_message
            )
        )


    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    model_messages = build_model_messages(
        chat_id=chat_id,
        user_query=user_message,
        current_content=user_message,
        exclude_message_id=message_id,
    )


    # --------------------------------------------------------
    # STREAM
    # --------------------------------------------------------

    return StreamingResponse(
        stream_chat(
            chat_id,
            model_messages,
        ),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# REGENERATE
# ============================================================

@app.post("/chats/{chat_id}/regenerate")
def regenerate_response(
    chat_id: str
):

    chat = get_chat(
        chat_id
    )

    if not chat:

        raise HTTPException(
            status_code=404,
            detail="Chat not found."
        )


    messages = get_messages(
        chat_id,
        limit=1000
    )

    if not messages:

        raise HTTPException(
            status_code=400,
            detail="No messages to regenerate."
        )


    # --------------------------------------------------------
    # DELETE LAST ASSISTANT
    # --------------------------------------------------------

    last_assistant = None

    for message in reversed(
        messages
    ):

        if message["role"] == "assistant":

            last_assistant = message

            break

    if last_assistant:

        delete_message(
            last_assistant["id"]
        )


    # --------------------------------------------------------
    # FIND LAST USER
    # --------------------------------------------------------

    messages = get_messages(
        chat_id,
        limit=1000
    )

    last_user = None

    for message in reversed(
        messages
    ):

        if message["role"] == "user":

            last_user = message

            break


    if not last_user:

        raise HTTPException(
            status_code=400,
            detail="No user message found."
        )


    user_message = last_user[
        "content"
    ]


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    model_messages = build_model_messages(
        chat_id=chat_id,
        user_query=user_message,
        current_content=None,
    )


    # --------------------------------------------------------
    # STREAM
    # --------------------------------------------------------

    return StreamingResponse(
        stream_chat(
            chat_id,
            model_messages,
        ),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# RENAME CHAT
# ============================================================

@app.put("/chats/{chat_id}")
def rename_chat(
    chat_id: str,
    request: RenameChatRequest,
):

    chat = get_chat(
        chat_id
    )

    if not chat:

        raise HTTPException(
            status_code=404,
            detail="Chat not found."
        )


    title = (
        request.title
        .strip()
    )

    if not title:
        title = "New Chat"


    update_chat_title(
        chat_id,
        title
    )


    updated_chat = get_chat(
        chat_id
    )

    return {
        "success": True,
        "chat": updated_chat,
    }


# ============================================================
# DELETE CHAT
# ============================================================

@app.delete("/chats/{chat_id}")
def delete_chat(
    chat_id: str
):

    chat = get_chat(
        chat_id
    )

    if not chat:

        raise HTTPException(
            status_code=404,
            detail="Chat not found."
        )


    conn = get_db()

    try:

        conn.execute(
            """
            DELETE FROM chats
            WHERE id = ?
            """,
            (chat_id,)
        )

        conn.commit()

    finally:
        conn.close()


    return {
        "success": True,
        "message": "Chat deleted successfully.",
        "chat_id": chat_id,
    }


# ============================================================
# CLEAR CHAT
# ============================================================

@app.post("/chats/{chat_id}/clear")
def clear_chat_messages(
    chat_id: str
):

    chat = get_chat(
        chat_id
    )

    if not chat:

        raise HTTPException(
            status_code=404,
            detail="Chat not found."
        )


    conn = get_db()

    try:

        conn.execute(
            """
            DELETE FROM messages
            WHERE chat_id = ?
            """,
            (chat_id,)
        )

        conn.execute(
            """
            UPDATE chats
            SET title = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                "New Chat",
                now_iso(),
                chat_id,
            )
        )

        conn.commit()

    finally:
        conn.close()


    return {
        "success": True,
        "message": "Conversation cleared.",
    }


# ============================================================
# MEMORY
# ============================================================

@app.get("/memory")
def get_memory(
    query: str = ""
):

    memories = search_memories(
        query,
        limit=50
    )

    return {
        "success": True,
        "memories": memories,
    }


# ============================================================
# DELETE MEMORY
# ============================================================

@app.delete("/memory/{memory_id}")
def delete_memory(
    memory_id: int
):

    conn = get_db()

    try:

        cursor = conn.execute(
            """
            DELETE FROM memories
            WHERE id = ?
            """,
            (memory_id,)
        )

        conn.commit()

        if cursor.rowcount == 0:

            raise HTTPException(
                status_code=404,
                detail="Memory not found."
            )

    finally:
        conn.close()


    return {
        "success": True,
        "message": "Memory deleted.",
    }


# ============================================================
# DELETE ALL MEMORY
# ============================================================

@app.delete("/memory")
def delete_all_memory():

    conn = get_db()

    try:

        conn.execute(
            "DELETE FROM memories"
        )

        conn.commit()

    finally:
        conn.close()


    return {
        "success": True,
        "message": "All memories deleted.",
    }


# ============================================================
# DOCUMENT LIST
# ============================================================

@app.get("/documents")
def list_documents():

    conn = get_db()

    try:

        rows = conn.execute(
            """
            SELECT *
            FROM documents
            ORDER BY created_at DESC
            """
        ).fetchall()

        return {
            "success": True,
            "documents": [
                dict(row)
                for row in rows
            ],
        }

    finally:
        conn.close()


# ============================================================
# REMOVE DOCUMENT FROM CHROMA
# ============================================================

def remove_document_from_rag(
    filename: str
):

    collection = get_rag_collection()

    if collection is None:
        return False

    try:

        collection.delete(
            where={
                "filename": filename
            }
        )

        print(
            f"🗑️ Removed RAG vectors: {filename}"
        )

        return True

    except Exception as error:

        print(
            f"⚠️ Could not remove "
            f"RAG document: {error}"
        )

        return False


# ============================================================
# PDF UPLOAD
# ============================================================

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    filename = Path(
        file.filename or ""
    ).name

    if not filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )


    if not filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )


    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    file_bytes = await file.read()

    max_size = 15 * 1024 * 1024

    if len(file_bytes) > max_size:

        raise HTTPException(
            status_code=413,
            detail=(
                "PDF is too large. "
                "Maximum size is 15 MB."
            )
        )


    if not file_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty."
        )


    # --------------------------------------------------------
    # PDF EXTRACTION
    # --------------------------------------------------------

    try:

        text = extract_pdf_text(
            file_bytes
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


    if not text:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not extract text from PDF. "
                "If this is a scanned PDF, OCR is required."
            )
        )


    # --------------------------------------------------------
    # CHUNK
    # --------------------------------------------------------

    chunks = chunk_text(
        text
    )

    if not chunks:

        raise HTTPException(
            status_code=400,
            detail="No readable text found in PDF."
        )


    # --------------------------------------------------------
    # CHROMA
    # --------------------------------------------------------

    collection = get_rag_collection()

    if collection is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Chroma RAG is unavailable.\n"
                f"Reason: {_rag_error or 'Unknown error'}\n\n"
                "Install:\n"
                "pip install chromadb sentence-transformers pypdf"
            )
        )


    # --------------------------------------------------------
    # REMOVE OLD VERSION
    # --------------------------------------------------------

    remove_document_from_rag(
        filename
    )


    conn = get_db()

    try:

        old_document = conn.execute(
            """
            SELECT file_path
            FROM documents
            WHERE filename = ?
            """,
            (filename,)
        ).fetchone()

        conn.execute(
            """
            DELETE FROM documents
            WHERE filename = ?
            """,
            (filename,)
        )

        conn.commit()

    finally:
        conn.close()


    # --------------------------------------------------------
    # SAVE FILE
    # --------------------------------------------------------

    file_path = (
        UPLOAD_DIR / filename
    )

    file_path.write_bytes(
        file_bytes
    )


    # --------------------------------------------------------
    # ADD TO CHROMA
    # --------------------------------------------------------

    ids = []
    documents = []
    metadatas = []

    try:

        for index, chunk in enumerate(
            chunks
        ):

            ids.append(
                f"{filename}-{uuid4().hex}"
            )

            documents.append(
                chunk
            )

            metadatas.append(
                {
                    "filename": filename,
                    "chunk_index": index,
                }
            )


        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        print(
            f"✅ Indexed PDF: {filename}"
        )

        print(
            f"   Chunks: {len(chunks)}"
        )


    except Exception as error:

        print(
            f"❌ Chroma indexing error: {error}"
        )

        # Remove partially indexed vectors
        try:

            collection.delete(
                ids=ids
            )

        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail=(
                "Document indexing failed: "
                f"{error}"
            )
        )


    # --------------------------------------------------------
    # SAVE DOCUMENT RECORD
    # --------------------------------------------------------

    conn = get_db()

    try:

        conn.execute(
            """
            INSERT INTO documents
            (
                filename,
                file_path,
                chunks,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                filename,
                str(file_path),
                len(chunks),
                now_iso(),
            )
        )

        conn.commit()

    except Exception as error:

        # If DB save fails, remove vectors
        try:

            collection.delete(
                where={
                    "filename": filename
                }
            )

        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not save document record: "
                f"{error}"
            )
        )

    finally:
        conn.close()


    # --------------------------------------------------------
    # DELETE OLD PHYSICAL FILE
    # --------------------------------------------------------

    if old_document:

        try:

            old_path = Path(
                old_document["file_path"]
            )

            if (
                old_path.exists()
                and old_path != file_path
            ):

                old_path.unlink()

        except Exception:
            pass


    return {
        "success": True,
        "message": (
            "PDF uploaded and indexed successfully."
        ),
        "document": {
            "filename": filename,
            "chunks": len(chunks),
        },
    }


# ============================================================
# DELETE DOCUMENT
# ============================================================

@app.delete("/documents/{filename}")
def delete_document(
    filename: str
):

    filename = Path(
        filename
    ).name


    # --------------------------------------------------------
    # REMOVE CHROMA
    # --------------------------------------------------------

    remove_document_from_rag(
        filename
    )


    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    conn = get_db()

    try:

        row = conn.execute(
            """
            SELECT file_path
            FROM documents
            WHERE filename = ?
            """,
            (filename,)
        ).fetchone()


        conn.execute(
            """
            DELETE FROM documents
            WHERE filename = ?
            """,
            (filename,)
        )

        conn.commit()

    finally:
        conn.close()


    # --------------------------------------------------------
    # PHYSICAL FILE
    # --------------------------------------------------------

    if row:

        try:

            path = Path(
                row["file_path"]
            )

            if path.exists():

                path.unlink()

        except Exception as error:

            print(
                f"⚠️ File deletion error: {error}"
            )


    return {
        "success": True,
        "message": "Document deleted successfully.",
        "filename": filename,
    }


# ============================================================
# IMAGE CHAT
# ============================================================

@app.post("/chats/{chat_id}/image")
async def image_chat(
    chat_id: str,
    file: UploadFile = File(...),
    question: str = Form(
        "Describe this image and explain what you can see."
    ),
):

    chat = get_chat(
        chat_id
    )

    if not chat:

        raise HTTPException(
            status_code=404,
            detail="Chat not found."
        )


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    filename = Path(
        file.filename or ""
    ).name

    extension = Path(
        filename
    ).suffix.lower()


    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }


    if extension not in mime_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Supported images: "
                "JPG, JPEG, PNG, WEBP."
            )
        )


    image_bytes = await file.read()


    if len(image_bytes) > 8 * 1024 * 1024:

        raise HTTPException(
            status_code=413,
            detail=(
                "Image is too large. "
                "Maximum size is 8 MB."
            )
        )


    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Image is empty."
        )


    # --------------------------------------------------------
    # BASE64
    # --------------------------------------------------------

    image_data = base64.b64encode(
        image_bytes
    ).decode("utf-8")


    image_url = (
        f"data:{mime_types[extension]};"
        f"base64,{image_data}"
    )


    question = (
        question.strip()
        or "Describe this image and explain what you can see."
    )


    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    user_content_for_db = (
        "[Image attached]\n"
        f"{question}"
    )


    message_id = save_message(
        chat_id,
        "user",
        user_content_for_db,
    )


    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    memory = extract_memory(
        question
    )

    if memory:

        memory_type, content = memory

        save_memory(
            memory_type,
            content
        )


    # --------------------------------------------------------
    # MODEL CONTEXT
    # --------------------------------------------------------

    model_messages = build_model_messages(
        chat_id=chat_id,
        user_query=question,
        current_content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                },
            },
            {
                "type": "text",
                "text": question,
            },
        ],
        exclude_message_id=message_id,
    )


    # --------------------------------------------------------
    # STREAM
    # --------------------------------------------------------

    return StreamingResponse(
        stream_chat(
            chat_id,
            model_messages,
        ),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# COMPLAINTS
# ============================================================

@app.post("/complaints")
def create_complaint(
    request: CreateComplaintRequest
):

    description = (
        request.description
        .strip()
    )

    if not description:

        raise HTTPException(
            status_code=400,
            detail="Complaint description cannot be empty."
        )


    timestamp = now_iso()

    conn = get_db()

    try:

        cursor = conn.execute(
            """
            INSERT INTO complaints
            (
                chat_id,
                description,
                created_at
            )
            VALUES (?, ?, ?)
            """,
            (
                request.chat_id,
                description,
                timestamp,
            )
        )

        conn.commit()

        complaint_id = cursor.lastrowid

    finally:
        conn.close()


    return {
        "success": True,
        "complaint": {
            "id": complaint_id,
            "chat_id": request.chat_id,
            "description": description,
            "created_at": timestamp,
        },
    }


# ============================================================
# RAG STATUS
# ============================================================

@app.get("/rag/status")
def rag_status():

    collection = get_rag_collection()

    if collection is None:

        return {
            "success": False,
            "available": False,
            "error": _rag_error,
            "embedding_model": EMBEDDING_MODEL,
        }


    try:

        count = collection.count()

    except Exception as error:

        return {
            "success": False,
            "available": False,
            "error": str(error),
        }


    return {
        "success": True,
        "available": True,
        "collection": "documents",
        "chunks": count,
        "embedding_model": EMBEDDING_MODEL,
    }


# ============================================================
# RAG TEST SEARCH
# ============================================================

@app.get("/rag/search")
def rag_search(
    query: str,
    limit: int = 5,
):

    query = query.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )


    results = search_documents(
        query,
        n_results=max(
            1,
            min(limit, 20)
        )
    )


    return {
        "success": True,
        "query": query,
        "count": len(results),
        "results": results,
    }


# ============================================================
# STATS
# ============================================================

@app.get("/stats")
def get_stats():

    conn = get_db()

    try:

        chats = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM chats
            """
        ).fetchone()["count"]


        messages = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM messages
            """
        ).fetchone()["count"]


        memories = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM memories
            """
        ).fetchone()["count"]


        documents = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM documents
            """
        ).fetchone()["count"]


        rag_chunks = 0

        collection = get_rag_collection()

        if collection is not None:

            try:

                rag_chunks = collection.count()

            except Exception:
                rag_chunks = 0


        return {
            "success": True,
            "stats": {
                "chats": chats,
                "messages": messages,
                "memories": memories,
                "documents": documents,
                "rag_chunks": rag_chunks,
            },
        }

    finally:
        conn.close()


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    print()
    print("=" * 70)
    print("🤖 AI ASSISTANT BACKEND")
    print("=" * 70)

    print(
        f"Model       : {MODEL_NAME}"
    )

    print(
        f"Database    : {DB_PATH}"
    )

    print(
        f"Upload Dir  : {UPLOAD_DIR}"
    )

    print(
        f"Chroma Dir  : {CHROMA_DIR}"
    )

    print(
        f"Embedding   : {EMBEDDING_MODEL}"
    )

    print("=" * 70)
    print()

    print(
        "🚀 Starting FastAPI server..."
    )

    print(
        "🌐 http://127.0.0.1:8000"
    )

    print(
        "📚 API docs: http://127.0.0.1:8000/docs"
    )

    print()

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000"
            )
        ),
        reload=os.getenv(
            "RELOAD",
            "true"
        ).lower() == "true",
    )