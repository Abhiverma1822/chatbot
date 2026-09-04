# ============================================================
# AI MEMORY + RAG CHATBOT API
# FastAPI Backend
# ============================================================

from pathlib import Path
import json
import sqlite3

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import StreamingResponse

from pydantic import BaseModel


# ============================================================
# IMPORT FROM MAIN.PY
# ============================================================

from main import (
    model,
    build_messages,
    clean_response,

    detect_memory,
    save_smart_memory,
    direct_response,

    create_chat_session,
    get_chat_session,
    list_chat_sessions,
    rename_chat_session,
    delete_chat_session,

    save_session_message,
    load_session_history,
    generate_chat_title,

    get_documents,
    remove_document,
    index_pdf,

    DB_PATH,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Memory + RAG API",
    description="AI Chatbot with Memory, RAG and Document Knowledge",
    version="4.0.0"
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
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):

    message: str


class CreateChatRequest(BaseModel):

    title: str = "New Chat"


class RenameChatRequest(BaseModel):

    title: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI Memory + RAG API is running",
        "version": "4.0.0",
        "status": "online"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "AI Memory + RAG API"
    }


# ============================================================
# CREATE NEW CHAT
# ============================================================

@app.post("/chats")
def create_chat(request: CreateChatRequest):

    title = request.title.strip()

    if not title:

        title = "New Chat"

    session_id = create_chat_session(
        title
    )

    chat = get_chat_session(
        session_id
    )

    return {
        "success": True,
        "chat": chat
    }


# ============================================================
# GET ALL CHATS
# ============================================================

@app.get("/chats")
def get_all_chats():

    chats = list_chat_sessions()

    return {
        "success": True,
        "chats": chats
    }


# ============================================================
# GET SINGLE CHAT
# ============================================================

@app.get("/chats/{session_id}")
def get_single_chat(session_id: int):

    chat = get_chat_session(
        session_id
    )

    if chat is None:

        raise HTTPException(
            status_code=404,
            detail="Chat session not found"
        )

    history = load_session_history(
        session_id,
        limit=100
    )

    messages = []

    for message in history:

        if message.type == "human":

            role = "user"

        elif message.type == "ai":

            role = "assistant"

        else:

            continue

        messages.append(
            {
                "role": role,
                "content": message.content
            }
        )

    return {
        "success": True,
        "chat": chat,
        "messages": messages
    }


# ============================================================
# SEND MESSAGE + STREAM AI RESPONSE
# ============================================================

@app.post("/chats/{session_id}/messages")
def send_message(
    session_id: int,
    request: ChatRequest
):

    user_input = request.message.strip()

    # --------------------------------------------------------
    # VALIDATE MESSAGE
    # --------------------------------------------------------

    if not user_input:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    # --------------------------------------------------------
    # CHECK CHAT
    # --------------------------------------------------------

    chat = get_chat_session(
        session_id
    )

    if chat is None:

        raise HTTPException(
            status_code=404,
            detail="Chat session not found"
        )

    # --------------------------------------------------------
    # LOAD PREVIOUS HISTORY
    # --------------------------------------------------------

    chat_history = load_session_history(
        session_id,
        limit=20
    )

    # --------------------------------------------------------
    # MEMORY DETECTION
    # --------------------------------------------------------

    memory = detect_memory(
        user_input
    )

    memory_saved = False

    if memory:

        try:

            save_smart_memory(
                memory
            )

            memory_saved = True

        except Exception as error:

            print(
                "Memory save error:",
                error
            )

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    save_session_message(
        session_id,
        "user",
        user_input
    )

    # --------------------------------------------------------
    # DIRECT RESPONSE
    #
    # Greetings / thanks etc.
    # --------------------------------------------------------

    direct_answer = direct_response(
        user_input
    )

    if direct_answer:

        save_session_message(
            session_id,
            "assistant",
            direct_answer
        )

        # --------------------------------------------
        # AUTO TITLE
        # --------------------------------------------

        if chat.get("title") == "New Chat":

            new_title = generate_chat_title(
                user_input
            )

            rename_chat_session(
                session_id,
                new_title
            )

        updated_chat = get_chat_session(
            session_id
        )

        def direct_generator():

            yield (
                json.dumps(
                    {
                        "type": "chunk",
                        "content": direct_answer
                    },
                    ensure_ascii=False
                )
                + "\n"
            )

            yield (
                json.dumps(
                    {
                        "type": "done",
                        "success": True,
                        "session_id": session_id,
                        "response": direct_answer,
                        "memory_saved": memory_saved,
                        "chat": updated_chat,
                        "streaming": True
                    },
                    ensure_ascii=False
                )
                + "\n"
            )

        return StreamingResponse(
            direct_generator(),
            media_type="application/x-ndjson"
        )

    # ========================================================
    # BUILD AI CONTEXT
    # ========================================================

    messages = build_messages(
        user_input,
        chat_history
    )

    # ========================================================
    # STREAM AI RESPONSE
    # ========================================================

    def generate():

        full_response = ""

        try:

            # ------------------------------------------------
            # MODEL STREAM
            # ------------------------------------------------

            for chunk in model.stream(
                messages
            ):

                text = ""

                if hasattr(
                    chunk,
                    "content"
                ):

                    text = chunk.content

                # --------------------------------------------
                # HANDLE LIST CONTENT
                # --------------------------------------------

                if isinstance(
                    text,
                    list
                ):

                    parts = []

                    for item in text:

                        if isinstance(
                            item,
                            dict
                        ):

                            parts.append(
                                str(
                                    item.get(
                                        "text",
                                        ""
                                    )
                                )
                            )

                        else:

                            parts.append(
                                str(item)
                            )

                    text = "".join(
                        parts
                    )

                else:

                    text = str(
                        text or ""
                    )

                # --------------------------------------------
                # EMPTY CHUNK
                # --------------------------------------------

                if not text:

                    continue

                # --------------------------------------------
                # STORE RESPONSE
                # --------------------------------------------

                full_response += text

                # --------------------------------------------
                # SEND CHUNK
                # --------------------------------------------

                yield (
                    json.dumps(
                        {
                            "type": "chunk",
                            "content": text
                        },
                        ensure_ascii=False
                    )
                    + "\n"
                )

            # ------------------------------------------------
            # CLEAN DEEPSEEK THINKING
            # ------------------------------------------------

            answer = clean_response(
                full_response
            )

            if not answer.strip():

                answer = (
                    "I couldn't generate "
                    "a response."
                )

            # ------------------------------------------------
            # SAVE ASSISTANT RESPONSE
            # ------------------------------------------------

            save_session_message(
                session_id,
                "assistant",
                answer
            )

            # ------------------------------------------------
            # AUTO TITLE
            # ------------------------------------------------

            if chat.get("title") == "New Chat":

                new_title = generate_chat_title(
                    user_input
                )

                rename_chat_session(
                    session_id,
                    new_title
                )

            # ------------------------------------------------
            # UPDATED CHAT
            # ------------------------------------------------

            updated_chat = get_chat_session(
                session_id
            )

            # ------------------------------------------------
            # DONE
            # ------------------------------------------------

            yield (
                json.dumps(
                    {
                        "type": "done",
                        "success": True,
                        "session_id": session_id,
                        "response": answer,
                        "memory_saved": memory_saved,
                        "chat": updated_chat,
                        "streaming": True
                    },
                    ensure_ascii=False
                )
                + "\n"
            )

        except Exception as error:

            print(
                "AI generation error:",
                error
            )

            yield (
                json.dumps(
                    {
                        "type": "error",
                        "success": False,
                        "message": (
                            "AI generation failed: "
                            + str(error)
                        )
                    },
                    ensure_ascii=False
                )
                + "\n"
            )

    # --------------------------------------------------------
    # RETURN STREAM
    # --------------------------------------------------------

    return StreamingResponse(

        generate(),

        media_type="application/x-ndjson",

        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no-cache"
        }
    )


# ============================================================
# REGENERATE LAST AI RESPONSE
# ============================================================

@app.post(
    "/chats/{session_id}/regenerate"
)
def regenerate_response(
    session_id: int
):

    # --------------------------------------------------------
    # CHECK SESSION
    # --------------------------------------------------------

    session = get_chat_session(
        session_id
    )

    if session is None:

        raise HTTPException(
            status_code=404,
            detail="Chat session not found"
        )

    # --------------------------------------------------------
    # LOAD HISTORY
    # --------------------------------------------------------

    history = load_session_history(
        session_id,
        limit=50
    )

    if not history:

        raise HTTPException(
            status_code=400,
            detail="No conversation history found"
        )

    # --------------------------------------------------------
    # LAST MESSAGE MUST BE AI
    # --------------------------------------------------------

    if history[-1].type != "ai":

        raise HTTPException(
            status_code=400,
            detail=(
                "There is no AI response "
                "to regenerate"
            )
        )

    # --------------------------------------------------------
    # REMOVE LAST AI FROM MEMORY
    # --------------------------------------------------------

    history_without_ai = list(
        history
    )

    history_without_ai.pop()

    # --------------------------------------------------------
    # LAST MESSAGE MUST BE USER
    # --------------------------------------------------------

    if (
        not history_without_ai
        or
        history_without_ai[-1].type != "human"
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid conversation history"
        )

    # --------------------------------------------------------
    # GET LAST USER MESSAGE
    # --------------------------------------------------------

    last_user_message = (
        history_without_ai[-1].content
    )

    # --------------------------------------------------------
    # PREVIOUS HISTORY
    # --------------------------------------------------------

    previous_history = (
        history_without_ai[:-1]
    )

    # --------------------------------------------------------
    # BUILD CONTEXT
    #
    # Memory + RAG + previous conversation
    # --------------------------------------------------------

    messages = build_messages(
        last_user_message,
        previous_history
    )

    # ========================================================
    # STREAM GENERATOR
    # ========================================================

    def generate():

        full_response = ""

        try:

            # ------------------------------------------------
            # MODEL STREAM
            # ------------------------------------------------

            for chunk in model.stream(
                messages
            ):

                text = ""

                if hasattr(
                    chunk,
                    "content"
                ):

                    text = chunk.content

                # --------------------------------------------
                # HANDLE LIST CONTENT
                # --------------------------------------------

                if isinstance(
                    text,
                    list
                ):

                    parts = []

                    for item in text:

                        if isinstance(
                            item,
                            dict
                        ):

                            parts.append(
                                str(
                                    item.get(
                                        "text",
                                        ""
                                    )
                                )
                            )

                        else:

                            parts.append(
                                str(item)
                            )

                    text = "".join(
                        parts
                    )

                else:

                    text = str(
                        text or ""
                    )

                # --------------------------------------------
                # IGNORE EMPTY
                # --------------------------------------------

                if not text:

                    continue

                # --------------------------------------------
                # STORE FULL RESPONSE
                # --------------------------------------------

                full_response += text

                # --------------------------------------------
                # SEND CHUNK
                # --------------------------------------------

                yield (
                    json.dumps(
                        {
                            "type": "chunk",
                            "content": text
                        },
                        ensure_ascii=False
                    )
                    + "\n"
                )

            # ------------------------------------------------
            # CLEAN RESPONSE
            # ------------------------------------------------

            answer = clean_response(
                full_response
            )

            if not answer.strip():

                answer = (
                    "I couldn't generate "
                    "a response."
                )

            # ------------------------------------------------
            # DELETE OLD AI RESPONSE
            # ------------------------------------------------

            conn = sqlite3.connect(
                DB_PATH
            )

            db_cursor = conn.cursor()

            db_cursor.execute(
                """
                DELETE FROM session_messages
                WHERE id = (
                    SELECT id
                    FROM session_messages
                    WHERE session_id = ?
                    AND role = 'assistant'
                    ORDER BY id DESC
                    LIMIT 1
                )
                """,
                (
                    session_id,
                )
            )

            conn.commit()

            conn.close()

            # ------------------------------------------------
            # SAVE NEW AI RESPONSE
            # ------------------------------------------------

            save_session_message(
                session_id,
                "assistant",
                answer
            )

            # ------------------------------------------------
            # GET UPDATED CHAT
            # ------------------------------------------------

            updated_session = (
                get_chat_session(
                    session_id
                )
            )

            # ------------------------------------------------
            # DONE
            # ------------------------------------------------

            yield (
                json.dumps(
                    {
                        "type": "done",
                        "success": True,
                        "session_id": session_id,
                        "response": answer,
                        "chat": updated_session,
                        "regenerated": True,
                        "streaming": True
                    },
                    ensure_ascii=False
                )
                + "\n"
            )

        except Exception as error:

            print(
                "Regeneration error:",
                error
            )

            yield (
                json.dumps(
                    {
                        "type": "error",
                        "success": False,
                        "message": (
                            "AI regeneration failed: "
                            + str(error)
                        )
                    },
                    ensure_ascii=False
                )
                + "\n"
            )

    # --------------------------------------------------------
    # RETURN STREAM
    # --------------------------------------------------------

    return StreamingResponse(

        generate(),

        media_type="application/x-ndjson",

        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no-cache"
        }
    )


# ============================================================
# RENAME CHAT
# ============================================================

@app.put("/chats/{session_id}")
def rename_chat(
    session_id: int,
    request: RenameChatRequest
):

    title = request.title.strip()

    if not title:

        raise HTTPException(
            status_code=400,
            detail="Chat title cannot be empty"
        )

    chat = get_chat_session(
        session_id
    )

    if chat is None:

        raise HTTPException(
            status_code=404,
            detail="Chat session not found"
        )

    rename_chat_session(
        session_id,
        title
    )

    updated_chat = get_chat_session(
        session_id
    )

    return {
        "success": True,
        "chat": updated_chat
    }


# ============================================================
# DELETE CHAT
# ============================================================

@app.delete("/chats/{session_id}")
def delete_chat(
    session_id: int
):

    chat = get_chat_session(
        session_id
    )

    if chat is None:

        raise HTTPException(
            status_code=404,
            detail="Chat session not found"
        )

    delete_chat_session(
        session_id
    )

    return {
        "success": True,
        "message": "Chat deleted successfully",
        "session_id": session_id
    }


# ============================================================
# OLD CHAT ENDPOINT
#
# Backward compatibility
# ============================================================

@app.post("/chat")
def old_chat_endpoint(
    request: ChatRequest
):

    user_input = request.message.strip()

    if not user_input:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    memory = detect_memory(
        user_input
    )

    memory_saved = False

    if memory:

        try:

            save_smart_memory(
                memory
            )

            memory_saved = True

        except Exception as error:

            print(
                "Memory error:",
                error
            )

    # --------------------------------------------------------
    # DIRECT RESPONSE
    # --------------------------------------------------------

    direct_answer = direct_response(
        user_input
    )

    if direct_answer:

        return {
            "success": True,
            "response": direct_answer,
            "memory_saved": memory_saved
        }

    # --------------------------------------------------------
    # BUILD MESSAGE
    # --------------------------------------------------------

    messages = build_messages(
        user_input,
        []
    )

    try:

        response = model.invoke(
            messages
        )

        answer = clean_response(
            response.content
        )

        return {
            "success": True,
            "response": answer,
            "memory_saved": memory_saved
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "AI generation failed: "
                + str(error)
            )
        )


# ============================================================
# MEMORY API
# ============================================================

@app.get("/memory")
def get_memory():

    try:

        conn = sqlite3.connect(
            DB_PATH
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                content,
                created_at
            FROM memories
            ORDER BY id DESC
            """
        )

        rows = cursor.fetchall()

        conn.close()

        memories = []

        for row in rows:

            memories.append(
                {
                    "id": row[0],
                    "content": row[1],
                    "created_at": row[2]
                }
            )

        return {
            "success": True,
            "memories": memories
        }

    except Exception as error:

        return {
            "success": False,
            "memories": [],
            "error": str(error)
        }


# ============================================================
# UPLOAD PDF DOCUMENT
# ============================================================

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected"
        )

    # --------------------------------------------------------
    # SAFE FILENAME
    # --------------------------------------------------------

    filename = Path(
        file.filename
    ).name

    # --------------------------------------------------------
    # ONLY PDF
    # --------------------------------------------------------

    if not filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # --------------------------------------------------------
    # DOCUMENT FOLDER
    # --------------------------------------------------------

    document_folder = Path(
        "documents"
    )

    document_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = (
        document_folder / filename
    )

    # --------------------------------------------------------
    # SAVE FILE
    # --------------------------------------------------------

    try:

        file_data = await file.read()

        with open(
            file_path,
            "wb"
        ) as output_file:

            output_file.write(
                file_data
            )

        # ----------------------------------------------------
        # INDEX PDF
        # ----------------------------------------------------

        result = index_pdf(
            str(file_path)
        )

        return {
            "success": True,
            "filename": filename,
            "message": "PDF uploaded and indexed successfully",
            "result": result
        }

    except Exception as error:

        if file_path.exists():

            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=(
                "PDF processing failed: "
                + str(error)
            )
        )


# ============================================================
# GET DOCUMENTS
# ============================================================

@app.get("/documents")
def documents():

    try:

        docs = get_documents()

        return {
            "success": True,
            "documents": docs
        }

    except Exception as error:

        return {
            "success": False,
            "documents": [],
            "error": str(error)
        }


# ============================================================
# DELETE DOCUMENT
# ============================================================

@app.delete(
    "/documents/{filename}"
)
def delete_document(
    filename: str
):

    safe_filename = Path(
        filename
    ).name

    try:

        result = remove_document(
            safe_filename
        )

        return {
            "success": True,
            "filename": safe_filename,
            "result": result
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Document deletion failed: "
                + str(error)
            )
        )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )