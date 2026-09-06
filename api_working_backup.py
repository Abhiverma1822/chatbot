import os
import base64
import json
import sqlite3
import uuid
import csv

from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Body,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from huggingface_hub import InferenceClient

from pypdf import PdfReader

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from memory import save_memory, search_memory


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# ENVIRONMENT
# ============================================================

ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        f"""
HF_TOKEN is missing.

Expected .env file:
{ENV_FILE}

Add your Hugging Face token to .env
"""
    )


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"

DB_FILE = BASE_DIR / "chat_memory.db"

DOCUMENTS_DIR = BASE_DIR / "documents"

DOCUMENTS_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    api_key=HF_TOKEN
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="DeepSeek AI Chatbot API",
    description="ChatGPT-style AI chatbot with Memory, Vision and RAG",
    version="3.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
- Never make up facts.
- If you do not know something, say so.
- Use Markdown when useful.
- When an image is provided, analyze it carefully.
- Use relevant conversation history when answering.
- Use relevant document context when provided.
- Do not reveal internal reasoning.
- Do not output <think> or </think>.
- Give only the final answer.

Language:

- Understand English, Hindi and Hinglish.
- Reply in the same language as the user when possible.
"""


# ============================================================
# DATABASE
# ============================================================

def get_db():

    connection = sqlite3.connect(
        DB_FILE
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():

    connection = get_db()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(chat_id) REFERENCES chats(id)
        )
        """
    )

    connection.commit()

    connection.close()


initialize_database()


# ============================================================
# TIME
# ============================================================

def current_time():

    return datetime.utcnow().isoformat()


# ============================================================
# CHAT HELPERS
# ============================================================

def create_chat_record(
    title="New Chat"
):

    chat_id = str(
        uuid.uuid4()
    )

    now = current_time()

    connection = get_db()

    cursor = connection.cursor()

    cursor.execute(
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
            title,
            now,
            now
        )
    )

    connection.commit()

    connection.close()

    return chat_id


def chat_exists(chat_id):

    connection = get_db()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id
        FROM chats
        WHERE id = ?
        """,
        (chat_id,)
    )

    result = cursor.fetchone()

    connection.close()

    return result is not None


def save_message(
    chat_id,
    role,
    content
):

    if not content:

        return

    connection = get_db()

    cursor = connection.cursor()

    now = current_time()

    cursor.execute(
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
            now
        )
    )

    cursor.execute(
        """
        UPDATE chats
        SET updated_at = ?
        WHERE id = ?
        """,
        (
            now,
            chat_id
        )
    )

    connection.commit()

    connection.close()


def get_chat_messages(
    chat_id
):

    connection = get_db()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            role,
            content,
            created_at
        FROM messages
        WHERE chat_id = ?
        ORDER BY id ASC
        """,
        (chat_id,)
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


def get_chat_title(
    chat_id
):

    connection = get_db()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT title
        FROM chats
        WHERE id = ?
        """,
        (chat_id,)
    )

    row = cursor.fetchone()

    connection.close()

    if row:

        return row["title"]

    return "New Chat"


# ============================================================
# DOCUMENT RAG
# ============================================================

print()
print("Loading document embedding model...")

document_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Document embedding model loaded.")


# ============================================================
# DOCUMENT CHROMA DATABASE
# ============================================================

DOCUMENT_VECTOR_STORE = Chroma(
    collection_name="chat_documents",
    embedding_function=document_embeddings,
    persist_directory=str(
        BASE_DIR / "chroma_documents"
    )
)


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(
    text,
    chunk_size=1000,
    chunk_overlap=150
):

    if not text:

        return []

    text = text.strip()

    if not text:

        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        chunk = chunk.strip()

        if chunk:

            chunks.append(chunk)

        if end >= text_length:

            break

        start = end - chunk_overlap

    return chunks


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(
    file_path
):

    reader = PdfReader(
        str(file_path)
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
                    f"""
[Page {page_number}]

{text}
"""
                )

        except Exception as error:

            print(
                f"PDF page {page_number} error:",
                error
            )

    return "\n".join(pages)


# ============================================================
# TEXT FILE EXTRACTION
# ============================================================

def extract_text_file(
    file_path
):

    try:

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception as error:

        print(
            "TEXT FILE ERROR:",
            error
        )

        return ""


# ============================================================
# CSV EXTRACTION
# ============================================================

def extract_csv_text(
    file_path
):

    try:

        rows = []

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore",
            newline=""
        ) as csv_file:

            reader = csv.reader(
                csv_file
            )

            for row in reader:

                rows.append(
                    " | ".join(row)
                )

        return "\n".join(rows)

    except Exception as error:

        print(
            "CSV ERROR:",
            error
        )

        return ""


# ============================================================
# EXTRACT DOCUMENT TEXT
# ============================================================

def extract_document_text(
    file_path
):

    extension = (
        file_path
        .suffix
        .lower()
    )

    if extension == ".pdf":

        return extract_pdf_text(
            file_path
        )

    if extension in [
        ".txt",
        ".md"
    ]:

        return extract_text_file(
            file_path
        )

    if extension == ".csv":

        return extract_csv_text(
            file_path
        )

    return ""


# ============================================================
# INDEX DOCUMENT
# ============================================================

def index_document(
    file_path
):

    filename = file_path.name

    print(
        f"Indexing document: {filename}"
    )

    text = extract_document_text(
        file_path
    )

    if not text.strip():

        raise ValueError(
            "Could not extract text from document."
        )

    chunks = split_text(
        text
    )

    if not chunks:

        raise ValueError(
            "No text chunks were created."
        )

    # --------------------------------------------------------
    # Remove previous version
    # --------------------------------------------------------

    try:

        DOCUMENT_VECTOR_STORE.delete(
            where={
                "source": filename
            }
        )

    except Exception:

        pass

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadatas = []

    for index, chunk in enumerate(
        chunks
    ):

        metadatas.append(
            {
                "source": filename,
                "chunk": index
            }
        )

    # --------------------------------------------------------
    # Add to Chroma
    # --------------------------------------------------------

    DOCUMENT_VECTOR_STORE.add_texts(
        texts=chunks,
        metadatas=metadatas
    )

    print(
        f"Indexed {len(chunks)} chunks from {filename}"
    )

    return len(chunks)


# ============================================================
# SEARCH DOCUMENTS
# ============================================================

def search_documents(
    query,
    k=4
):

    try:

        results = (
            DOCUMENT_VECTOR_STORE
            .similarity_search(
                query,
                k=k
            )
        )

        return results

    except Exception as error:

        print(
            "DOCUMENT SEARCH ERROR:",
            error
        )

        return []


# ============================================================
# BUILD DOCUMENT CONTEXT
# ============================================================

def get_document_context(
    query
):

    results = search_documents(
        query,
        k=4
    )

    if not results:

        return ""

    context_parts = []

    for result in results:

        source = result.metadata.get(
            "source",
            "Unknown document"
        )

        context_parts.append(
            f"""
Source: {source}

{result.page_content}
"""
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# AI MESSAGE BUILDER
# ============================================================

def build_messages(
    chat_id,
    user_message
):

    history = get_chat_messages(
        chat_id
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # --------------------------------------------------------
    # LONG TERM MEMORY
    # --------------------------------------------------------

    try:

        memories = search_memory(
            user_message,
            k=3
        )

        if memories:

            memory_text = "\n\n".join(
                [
                    item.page_content
                    for item in memories
                ]
            )

            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Relevant long-term memory:\n\n"
                        f"{memory_text}"
                    )
                }
            )

    except Exception as error:

        print(
            "MEMORY SEARCH ERROR:",
            error
        )

    # --------------------------------------------------------
    # DOCUMENT RAG
    # --------------------------------------------------------

    try:

        document_context = (
            get_document_context(
                user_message
            )
        )

        if document_context:

            messages.append(
                {
                    "role": "system",
                    "content": f"""
Relevant information retrieved from uploaded documents:

{document_context}

Instructions:
- Use this document information when it is relevant.
- Prefer the document information for questions about uploaded files.
- If the answer is not present in the documents, say that it is not available in the uploaded documents.
"""
                }
            )

    except Exception as error:

        print(
            "RAG CONTEXT ERROR:",
            error
        )

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    for item in history:

        role = item["role"]

        content = item["content"]

        if role in [
            "user",
            "assistant"
        ]:

            messages.append(
                {
                    "role": role,
                    "content": content
                }
            )

    return messages


# ============================================================
# STREAM RESPONSE
# ============================================================

def generate_stream(
    chat_id,
    user_message
):

    try:

        messages = build_messages(
            chat_id,
            user_message
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
            stream=True
        )

        full_response = ""

        for chunk in response:

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

                if content:

                    full_response += content

                    yield json.dumps(
                        {
                            "type": "chunk",
                            "content": content
                        }
                    ) + "\n"

            except Exception as error:

                print(
                    "STREAM CHUNK ERROR:",
                    error
                )

        # ----------------------------------------------------
        # SAVE ASSISTANT MESSAGE
        # ----------------------------------------------------

        if full_response.strip():

            save_message(
                chat_id,
                "assistant",
                full_response
            )

            # ------------------------------------------------
            # LONG TERM MEMORY
            # ------------------------------------------------

            try:

                save_memory(
                    f"""
User asked:
{user_message}

Assistant answered:
{full_response}
"""
                )

            except Exception as error:

                print(
                    "MEMORY SAVE ERROR:",
                    error
                )

        # ----------------------------------------------------
        # DONE
        # ----------------------------------------------------

        yield json.dumps(
            {
                "type": "done",
                "response": full_response
            }
        ) + "\n"

    except Exception as error:

        print(
            "STREAM ERROR:",
            error
        )

        yield json.dumps(
            {
                "type": "error",
                "message": str(error)
            }
        ) + "\n"


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def home():

    return {
        "status": "online",
        "message": "DeepSeek AI Chatbot API is running",
        "model": MODEL_NAME,
        "features": [
            "text",
            "vision",
            "streaming",
            "chat_history",
            "long_term_memory",
            "document_rag"
        ]
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "database": "connected",
        "rag": "enabled",
        "memory": "enabled"
    }


# ============================================================
# CREATE CHAT
# ============================================================

@app.post("/chats")
async def create_chat(
    data: dict = Body(default={})
):

    title = data.get(
        "title",
        "New Chat"
    )

    if not title:

        title = "New Chat"

    chat_id = create_chat_record(
        title
    )

    return {
        "success": True,
        "chat": {
            "id": chat_id,
            "title": title
        }
    }


# ============================================================
# GET ALL CHATS
# ============================================================

@app.get("/chats")
async def get_chats():

    connection = get_db()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            title,
            created_at,
            updated_at
        FROM chats
        ORDER BY updated_at DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return {
        "success": True,
        "chats": [
            dict(row)
            for row in rows
        ]
    }


# ============================================================
# GET SINGLE CHAT
# ============================================================

@app.get("/chats/{chat_id}")
async def get_chat(
    chat_id: str
):

    if not chat_exists(
        chat_id
    ):

        raise HTTPException(
            status_code=404,
            detail="Chat not found."
        )

    return {
        "success": True,
        "chat": {
            "id": chat_id,
            "title": get_chat_title(
                chat_id
            )
        },
        "messages": get_chat_messages(
            chat_id
        )
    }


# ============================================================
# RENAME CHAT
# ============================================================

@app.put("/chats/{chat_id}")
async def rename_chat(
    chat_id: str,
    data: dict = Body(...)
):

    if not chat_exists(
        chat_id
    ):

        raise HTTPException(
            status_code=404,
            detail="Chat not found."
        )

    title = str(
        data.get(
            "title",
            "New Chat"
        )
    ).strip()

    if not title:

        title = "New Chat"

    connection = get_db()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE chats
        SET
            title = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            title,
            current_time(),
            chat_id
        )
    )

    connection.commit()

    connection.close()

    return {
        "success": True,
        "title": title
    }


# ============================================================
# DELETE CHAT
# ============================================================

@app.delete("/chats/{chat_id}")
async def delete_chat(
    chat_id: str
):

    if not chat_exists(
        chat_id
    ):

        raise HTTPException(
            status_code=404,
            detail="Chat not found."
        )

    connection = get_db()

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM messages
        WHERE chat_id = ?
        """,
        (chat_id,)
    )

    cursor.execute(
        """
        DELETE FROM chats
        WHERE id = ?
        """,
        (chat_id,)
    )

    connection.commit()

    connection.close()

    return {
        "success": True,
        "message": "Chat deleted successfully."
    }


# ============================================================
# SEND MESSAGE
# ============================================================

@app.post("/chats/{chat_id}/messages")
async def send_message(
    chat_id: str,
    data: dict = Body(...)
):

    if not chat_exists(
        chat_id
    ):

        raise HTTPException(
            status_code=404,
            detail="Chat not found."
        )

    message = str(
        data.get(
            "message",
            ""
        )
    ).strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    save_message(
        chat_id,
        "user",
        message
    )

    # --------------------------------------------------------
    # AUTO TITLE
    # --------------------------------------------------------

    current_title = get_chat_title(
        chat_id
    )

    if current_title == "New Chat":

        title = message[:40]

        connection = get_db()

        cursor = connection.cursor()

        cursor.execute(
            """
            UPDATE chats
            SET
                title = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                title,
                current_time(),
                chat_id
            )
        )

        connection.commit()

        connection.close()

    # --------------------------------------------------------
    # STREAM
    # --------------------------------------------------------

    return StreamingResponse(
        generate_stream(
            chat_id,
            message
        ),
        media_type="application/x-ndjson"
    )


# ============================================================
# REGENERATE
# ============================================================

@app.post("/chats/{chat_id}/regenerate")
async def regenerate_response(
    chat_id: str
):

    if not chat_exists(
        chat_id
    ):

        raise HTTPException(
            status_code=404,
            detail="Chat not found."
        )

    connection = get_db()

    cursor = connection.cursor()

    # --------------------------------------------------------
    # DELETE LAST ASSISTANT
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT id
        FROM messages
        WHERE
            chat_id = ?
            AND role = 'assistant'
        ORDER BY id DESC
        LIMIT 1
        """,
        (chat_id,)
    )

    assistant_row = cursor.fetchone()

    if assistant_row:

        cursor.execute(
            """
            DELETE FROM messages
            WHERE id = ?
            """,
            (
                assistant_row["id"],
            )
        )

    # --------------------------------------------------------
    # GET LAST USER MESSAGE
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT content
        FROM messages
        WHERE
            chat_id = ?
            AND role = 'user'
        ORDER BY id DESC
        LIMIT 1
        """,
        (chat_id,)
    )

    user_row = cursor.fetchone()

    connection.commit()

    connection.close()

    if not user_row:

        raise HTTPException(
            status_code=400,
            detail=(
                "No user message available "
                "to regenerate."
            )
        )

    user_message = user_row[
        "content"
    ]

    return StreamingResponse(
        generate_stream(
            chat_id,
            user_message
        ),
        media_type="application/x-ndjson"
    )


# ============================================================
# OLD /CHAT ENDPOINT
# ============================================================

@app.post("/chat")
async def old_chat(
    message: str = Form(...)
):

    message = message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            max_tokens=1024,
            temperature=0.3
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        return {
            "success": True,
            "model": MODEL_NAME,
            "message": message,
            "response": answer
        }

    except Exception as error:

        print(
            "TEXT CHAT ERROR:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=f"AI request failed: {str(error)}"
        )


# ============================================================
# VISION
# ============================================================

@app.post("/vision")
async def vision(
    message: str = Form(...),
    image: UploadFile = File(...)
):

    message = message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    if not image.content_type:

        raise HTTPException(
            status_code=400,
            detail="Could not determine image type."
        )

    if not image.content_type.startswith(
        "image/"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only image files are allowed."
        )

    try:

        image_bytes = await image.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        encoded_image = (
            base64
            .b64encode(image_bytes)
            .decode("utf-8")
        )

        mime_type = image.content_type

        image_url = (
            f"data:{mime_type};base64,"
            f"{encoded_image}"
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": message
                    }
                ]
            }
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1024,
            temperature=0.3
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        if not answer:

            answer = (
                "Sorry, I could not "
                "analyze the image."
            )

        return {
            "success": True,
            "model": MODEL_NAME,
            "filename": image.filename,
            "question": message,
            "response": answer
        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            "VISION ERROR:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Image AI request failed: "
                f"{str(error)}"
            )
        )


# ============================================================
# MEMORY
# ============================================================

@app.get("/memory")
async def get_memory():

    try:

        # Chroma similarity search requires
        # a meaningful query.
        memories = search_memory(
            "user assistant conversation",
            k=20
        )

        result = []

        for memory in memories:

            result.append(
                {
                    "content": memory.page_content,
                    "metadata": memory.metadata
                }
            )

        return {
            "success": True,
            "memories": result
        }

    except Exception as error:

        print(
            "MEMORY ERROR:",
            error
        )

        return {
            "success": True,
            "memories": []
        }


# ============================================================
# GET DOCUMENTS
# ============================================================

@app.get("/documents")
async def get_documents():

    documents = []

    for file in DOCUMENTS_DIR.iterdir():

        if file.is_file():

            documents.append(
                {
                    "filename": file.name,
                    "size": file.stat().st_size
                }
            )

    return {
        "success": True,
        "documents": documents
    }


# ============================================================
# UPLOAD + INDEX DOCUMENT
# ============================================================

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is missing."
        )

    extension = (
        Path(file.filename)
        .suffix
        .lower()
    )

    allowed_extensions = {
        ".pdf",
        ".txt",
        ".md",
        ".csv"
    }

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF, TXT, MD and CSV "
                "files are supported."
            )
        )

    try:

        # ----------------------------------------------------
        # Safe filename
        # ----------------------------------------------------

        safe_filename = Path(
            file.filename
        ).name

        destination = (
            DOCUMENTS_DIR /
            safe_filename
        )

        # ----------------------------------------------------
        # Save file
        # ----------------------------------------------------

        file_bytes = await file.read()

        if not file_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        with open(
            destination,
            "wb"
        ) as output_file:

            output_file.write(
                file_bytes
            )

        # ----------------------------------------------------
        # INDEX DOCUMENT
        # ----------------------------------------------------

        chunk_count = index_document(
            destination
        )

        return {
            "success": True,
            "filename": safe_filename,
            "chunks": chunk_count,
            "message": (
                "Document uploaded and "
                "indexed successfully."
            )
        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            "DOCUMENT UPLOAD ERROR:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Document processing failed: "
                f"{str(error)}"
            )
        )


# ============================================================
# DELETE DOCUMENT
# ============================================================

@app.delete("/documents/{filename}")
async def delete_document(
    filename: str
):

    safe_filename = Path(
        filename
    ).name

    file_path = (
        DOCUMENTS_DIR /
        safe_filename
    )

    if not file_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    try:

        # ----------------------------------------------------
        # Delete file
        # ----------------------------------------------------

        file_path.unlink()

        # ----------------------------------------------------
        # Delete vectors
        # ----------------------------------------------------

        try:

            DOCUMENT_VECTOR_STORE.delete(
                where={
                    "source": safe_filename
                }
            )

        except Exception as error:

            print(
                "CHROMA DELETE WARNING:",
                error
            )

        return {
            "success": True,
            "message": (
                "Document and its "
                "vectors deleted successfully."
            )
        }

    except Exception as error:

        print(
            "DOCUMENT DELETE ERROR:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Delete failed: "
                f"{str(error)}"
            )
        )


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )