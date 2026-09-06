// ============================================================
// AI MEMORY + RAG CHATBOT
// COMPLETE FRONTEND - FIXED VERSION
// Backend: FastAPI + DeepSeek + Chroma RAG + Memory
// ============================================================

const API_URL = "http://127.0.0.1:8000";

// ============================================================
// GLOBAL STATE
// ============================================================

let currentSessionId =
    localStorage.getItem("currentSessionId") || null;

let isGenerating = false;
let currentAbortController = null;

// ============================================================
// DOM ELEMENTS
// ============================================================

const chatMessages = document.getElementById("chatMessages");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const stopBtn = document.getElementById("stopBtn");
const newChatBtn = document.getElementById("newChatBtn");
const clearBtn = document.getElementById("clearBtn");

const memoryBtn = document.getElementById("memoryBtn");
const closeMemoryBtn = document.getElementById("closeMemoryBtn");
const memoryPanel = document.getElementById("memoryPanel");
const memoryContent = document.getElementById("memoryContent");

const pdfInput = document.getElementById("pdfInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");
const documentList = document.getElementById("documentList");

const chatHistory = document.getElementById("chatHistory");
const statusText = document.getElementById("status");

// ============================================================
// SAFE HTML
// ============================================================

function escapeHTML(text) {
    const div = document.createElement("div");
    div.textContent = String(text ?? "");
    return div.innerHTML;
}

// ============================================================
// CLEAN THINKING TAGS
// ============================================================

function cleanThinking(text) {
    if (!text) {
        return "";
    }

    let cleaned = String(text);

    // Complete <think>...</think>
    cleaned = cleaned.replace(
        /<think>[\s\S]*?<\/think>/gi,
        ""
    );

    // Incomplete <think> at end
    cleaned = cleaned.replace(
        /<think>[\s\S]*$/gi,
        ""
    );

    // Remove standalone tags
    cleaned = cleaned.replace(
        /<\/?think>/gi,
        ""
    );

    return cleaned.trim();
}

// ============================================================
// MARKDOWN FORMATTER
// ============================================================

function formatMarkdown(text) {
    if (!text) {
        return "";
    }

    let source = cleanThinking(text);

    if (!source) {
        return "";
    }

    // --------------------------------------------------------
    // Extract code blocks first
    // --------------------------------------------------------

    const codeBlocks = [];

    source = source.replace(
        /```([\w+#.-]*)\r?\n?([\s\S]*?)```/g,
        function (_, language, code) {
            const index = codeBlocks.length;

            code = code.replace(/\n$/, "");

            codeBlocks.push({
                language: language || "text",
                code: code
            });

            return `@@CODEBLOCK_${index}@@`;
        }
    );

    // --------------------------------------------------------
    // Escape HTML
    // --------------------------------------------------------

    source = escapeHTML(source);

    // --------------------------------------------------------
    // Headings
    // --------------------------------------------------------

    source = source.replace(
        /^### (.+)$/gm,
        "<h3>$1</h3>"
    );

    source = source.replace(
        /^## (.+)$/gm,
        "<h2>$1</h2>"
    );

    source = source.replace(
        /^# (.+)$/gm,
        "<h1>$1</h1>"
    );

    // --------------------------------------------------------
    // Bold
    // --------------------------------------------------------

    source = source.replace(
        /\*\*(.+?)\*\*/g,
        "<strong>$1</strong>"
    );

    // --------------------------------------------------------
    // Italic
    // --------------------------------------------------------

    source = source.replace(
        /(^|[^*])\*([^*\n]+)\*(?!\*)/g,
        "$1<em>$2</em>"
    );

    // --------------------------------------------------------
    // Inline code
    // --------------------------------------------------------

    source = source.replace(
        /`([^`\n]+)`/g,
        "<code>$1</code>"
    );

    // --------------------------------------------------------
    // Links
    // --------------------------------------------------------

    source = source.replace(
        /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );

    // --------------------------------------------------------
    // Unordered lists
    // --------------------------------------------------------

    source = source.replace(
        /(^|\n)[-*]\s+(.+)/g,
        "$1<li>$2</li>"
    );

    source = source.replace(
        /((?:<li>.*?<\/li>\s*)+)/g,
        function (list) {
            return `<ul>${list}</ul>`;
        }
    );

    // --------------------------------------------------------
    // Ordered lists
    // --------------------------------------------------------

    source = source.replace(
        /(^|\n)\d+\.\s+(.+)/g,
        "$1<oli>$2</oli>"
    );

    source = source.replace(
        /((?:<oli>.*?<\/oli>\s*)+)/g,
        function (list) {
            return (
                "<ol>" +
                list
                    .replace(/<oli>/g, "<li>")
                    .replace(/<\/oli>/g, "</li>") +
                "</ol>"
            );
        }
    );

    // --------------------------------------------------------
    // Horizontal rule
    // --------------------------------------------------------

    source = source.replace(
        /^---$/gm,
        "<hr>"
    );

    // --------------------------------------------------------
    // New lines
    // --------------------------------------------------------

    source = source.replace(
        /\n/g,
        "<br>"
    );

    // --------------------------------------------------------
    // Restore code blocks
    // --------------------------------------------------------

    codeBlocks.forEach(function (block, index) {
        const escapedCode = escapeHTML(block.code);
        const language = block.language || "text";

        const codeHTML = `
            <div class="code-wrapper">

                <div class="code-header">

                    <span class="code-language">
                        ${escapeHTML(language)}
                    </span>

                    <button
                        class="copy-code-btn"
                        type="button"
                        onclick="copyCode(this)"
                        data-code="${encodeURIComponent(block.code)}"
                    >
                        Copy
                    </button>

                </div>

                <pre><code class="language-${escapeHTML(language)}">${escapedCode}</code></pre>

            </div>
        `;

        source = source.replace(
            `@@CODEBLOCK_${index}@@`,
            codeHTML
        );
    });

    return source;
}

// ============================================================
// FORMAT MESSAGE
// ============================================================

function formatMessage(text) {
    return formatMarkdown(
        cleanThinking(text)
    );
}

// ============================================================
// SYNTAX HIGHLIGHTING
// ============================================================

function applySyntaxHighlighting() {
    if (typeof hljs === "undefined") {
        return;
    }

    document
        .querySelectorAll(".code-wrapper pre code")
        .forEach(function (block) {

            try {
                if (!block.dataset.highlighted) {
                    hljs.highlightElement(block);
                }
            } catch (error) {
                console.warn(
                    "Highlight error:",
                    error
                );
            }

        });
}

// ============================================================
// COPY CODE
// ============================================================

async function copyCode(button) {

    try {

        const encodedCode =
            button.getAttribute("data-code") || "";

        const code =
            decodeURIComponent(encodedCode);

        await navigator.clipboard.writeText(code);

        const oldText =
            button.textContent;

        button.textContent = "Copied!";

        setTimeout(function () {
            button.textContent = oldText;
        }, 1500);

    } catch (error) {

        console.error(
            "Copy failed:",
            error
        );

        button.textContent = "Failed";

        setTimeout(function () {
            button.textContent = "Copy";
        }, 1500);
    }
}

// ============================================================
// SCROLL
// ============================================================

function scrollToBottom() {

    if (!chatMessages) {
        return;
    }

    chatMessages.scrollTop =
        chatMessages.scrollHeight;
}

// ============================================================
// WELCOME SCREEN
// ============================================================

function showWelcome() {

    if (!chatMessages) {
        return;
    }

    chatMessages.innerHTML = `
        <div class="welcome-screen">

            <div class="welcome-icon">
                🤖
            </div>

            <h1>
                AI Memory + RAG Assistant
            </h1>

            <p>
                Ask me anything. I can remember
                important information, search
                uploaded documents and generate
                AI responses.
            </p>

            <div class="suggestions">

                <button class="suggestion-btn">
                    Explain machine learning
                </button>

                <button class="suggestion-btn">
                    What is RAG?
                </button>

                <button class="suggestion-btn">
                    Explain Python in simple words
                </button>

                <button class="suggestion-btn">
                    What is Generative AI?
                </button>

            </div>

        </div>
    `;

    attachSuggestionEvents();
}

// ============================================================
// ADD MESSAGE
// ============================================================

function addMessage(
    message,
    role,
    showActions = false
) {

    if (!chatMessages) {
        return null;
    }

    const messageDiv =
        document.createElement("div");

    messageDiv.className =
        `message ${role}-message`;

    const avatar =
        document.createElement("div");

    avatar.className =
        "message-avatar";

    avatar.textContent =
        role === "user"
            ? "👤"
            : "🤖";

    const contentWrapper =
        document.createElement("div");

    contentWrapper.className =
        "message-content-wrapper";

    const content =
        document.createElement("div");

    content.className =
        "message-content";

    if (role === "user") {

        content.textContent =
            message;

    } else {

        content.innerHTML =
            formatMessage(message);
    }

    contentWrapper.appendChild(content);

    if (
        role === "assistant" &&
        showActions
    ) {

        addRegenerateButton(
            messageDiv,
            contentWrapper
        );
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentWrapper);

    chatMessages.appendChild(messageDiv);

    applySyntaxHighlighting();
    scrollToBottom();

    return messageDiv;
}

// ============================================================
// ADD REGENERATE BUTTON
// ============================================================

function addRegenerateButton(
    messageDiv,
    wrapper
) {

    const actions =
        document.createElement("div");

    actions.className =
        "message-actions";

    const button =
        document.createElement("button");

    button.type = "button";

    button.className =
        "regenerate-btn";

    button.textContent =
        "↻ Regenerate";

    button.addEventListener(
        "click",
        function () {

            regenerateResponse(
                messageDiv
            );

        }
    );

    actions.appendChild(button);

    wrapper.appendChild(actions);
}

// ============================================================
// SHOW REGENERATE BUTTON
// ============================================================

function showRegenerateButton(messageDiv) {

    if (!messageDiv) {
        return;
    }

    const wrapper =
        messageDiv.querySelector(
            ".message-content-wrapper"
        );

    if (!wrapper) {
        return;
    }

    const oldActions =
        messageDiv.querySelector(
            ".message-actions"
        );

    if (oldActions) {
        oldActions.remove();
    }

    addRegenerateButton(
        messageDiv,
        wrapper
    );
}

// ============================================================
// HIDE REGENERATE BUTTONS
// ============================================================

function hideAllRegenerateButtons() {

    document
        .querySelectorAll(".message-actions")
        .forEach(function (element) {

            element.remove();

        });
}

// ============================================================
// STREAMING MESSAGE
// ============================================================

function createStreamingMessage() {

    if (!chatMessages) {
        return null;
    }

    const messageDiv =
        document.createElement("div");

    messageDiv.className =
        "message assistant-message streaming-message";

    const avatar =
        document.createElement("div");

    avatar.className =
        "message-avatar";

    avatar.textContent =
        "🤖";

    const wrapper =
        document.createElement("div");

    wrapper.className =
        "message-content-wrapper";

    const content =
        document.createElement("div");

    content.className =
        "message-content";

    content.innerHTML = `
        <span class="loading-dots">
            <span></span>
            <span></span>
            <span></span>
        </span>
    `;

    wrapper.appendChild(content);

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(wrapper);

    chatMessages.appendChild(messageDiv);

    scrollToBottom();

    return {
        messageDiv,
        content
    };
}

// ============================================================
// PROCESS STREAM LINE
// ============================================================

function processStreamLine(
    line,
    state
) {

    if (!line || !line.trim()) {
        return;
    }

    let data;

    try {

        data =
            JSON.parse(line);

    } catch (error) {

        console.warn(
            "Invalid stream JSON:",
            line
        );

        return;
    }

    // --------------------------------------------------------
    // CHUNK
    // --------------------------------------------------------

    if (data.type === "chunk") {

        const chunk =
            data.content || "";

        state.fullResponse +=
            chunk;

        state.aiContent.innerHTML =
            formatMessage(
                state.fullResponse
            );

        applySyntaxHighlighting();
        scrollToBottom();

        return;
    }

    // --------------------------------------------------------
    // DONE
    // --------------------------------------------------------

    if (data.type === "done") {

        state.completed = true;

        const finalResponse =
            data.response ||
            state.fullResponse ||
            "";

        state.fullResponse =
            finalResponse;

        state.aiContent.innerHTML =
            formatMessage(
                finalResponse
            );

        applySyntaxHighlighting();

        showRegenerateButton(
            state.messageDiv
        );

        // UUID MUST remain string
        if (
            data.chat &&
            data.chat.id
        ) {

            currentSessionId =
                String(data.chat.id);

            localStorage.setItem(
                "currentSessionId",
                currentSessionId
            );
        }

        if (statusText) {

            statusText.textContent =
                "● Online";
        }

        scrollToBottom();

        return;
    }

    // --------------------------------------------------------
    // ERROR
    // --------------------------------------------------------

    if (data.type === "error") {

        state.streamError =
            data.message ||
            "AI generation failed.";

        state.aiContent.innerHTML = `
            <div class="error-message">
                ${escapeHTML(
            state.streamError
        )}
            </div>
        `;

        return;
    }
}

// ============================================================
// READ STREAM
// ============================================================

async function readStreamingResponse(
    response,
    state
) {

    if (!response.body) {

        throw new Error(
            "Streaming response body is unavailable."
        );
    }

    const reader =
        response.body.getReader();

    const decoder =
        new TextDecoder("utf-8");

    let buffer = "";

    while (true) {

        const result =
            await reader.read();

        if (result.done) {
            break;
        }

        buffer +=
            decoder.decode(
                result.value,
                {
                    stream: true
                }
            );

        const lines =
            buffer.split("\n");

        buffer =
            lines.pop() || "";

        for (const line of lines) {

            processStreamLine(
                line,
                state
            );
        }
    }

    if (buffer.trim()) {

        processStreamLine(
            buffer,
            state
        );
    }
}

// ============================================================
// SEND MESSAGE
// ============================================================

async function sendMessage() {

    if (isGenerating) {
        return;
    }

    if (!userInput) {
        return;
    }

    const message =
        userInput.value.trim();

    if (!message) {
        return;
    }

    // --------------------------------------------------------
    // Create chat if required
    // --------------------------------------------------------

    if (!currentSessionId) {

        const chat =
            await createNewChat();

        if (!chat || !chat.id) {

            alert(
                "Unable to create chat session."
            );

            return;
        }
    }

    if (!currentSessionId) {

        alert(
            "Unable to create chat session."
        );

        return;
    }

    // --------------------------------------------------------
    // Remove welcome
    // --------------------------------------------------------

    const welcome =
        chatMessages
            ? chatMessages.querySelector(
                ".welcome-screen"
            )
            : null;

    if (welcome) {
        welcome.remove();
    }

    hideAllRegenerateButtons();

    // --------------------------------------------------------
    // User message
    // --------------------------------------------------------

    addMessage(
        message,
        "user"
    );

    userInput.value = "";

    autoResize();

    // --------------------------------------------------------
    // State
    // --------------------------------------------------------

    isGenerating = true;

    currentAbortController =
        new AbortController();

    if (sendBtn) {
        sendBtn.disabled = true;
    }

    if (stopBtn) {
        stopBtn.classList.remove("hidden");
    }

    if (statusText) {
        statusText.textContent =
            "● Thinking...";
    }

    // --------------------------------------------------------
    // Streaming AI message
    // --------------------------------------------------------

    const streaming =
        createStreamingMessage();

    if (!streaming) {

        isGenerating = false;

        return;
    }

    const state = {

        aiContent:
            streaming.content,

        messageDiv:
            streaming.messageDiv,

        fullResponse:
            "",

        completed:
            false,

        streamError:
            null
    };

    try {

        const response =
            await fetch(
                `${API_URL}/chats/${encodeURIComponent(currentSessionId)}/messages`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        message: message
                    }),

                    signal:
                        currentAbortController.signal
                }
            );

        // ----------------------------------------------------
        // SERVER ERROR
        // ----------------------------------------------------

        if (!response.ok) {

            let errorMessage =
                `Server error: ${response.status}`;

            try {

                const errorData =
                    await response.json();

                errorMessage =
                    errorData.detail ||
                    errorData.message ||
                    errorMessage;

            } catch (_) { }

            throw new Error(
                errorMessage
            );
        }

        await readStreamingResponse(
            response,
            state
        );

        if (
            !state.completed &&
            state.fullResponse
        ) {

            state.aiContent.innerHTML =
                formatMessage(
                    state.fullResponse
                );

            showRegenerateButton(
                state.messageDiv
            );
        }

        await loadChatHistoryList();

    } catch (error) {

        if (
            error.name ===
            "AbortError"
        ) {

            if (statusText) {

                statusText.textContent =
                    "● Generation stopped";
            }

            state.aiContent.innerHTML =
                formatMessage(
                    state.fullResponse
                );

            if (state.fullResponse) {

                const notice =
                    document.createElement("div");

                notice.className =
                    "generation-stopped";

                notice.textContent =
                    "Generation stopped.";

                state.aiContent.appendChild(
                    notice
                );
            }

        } else {

            console.error(
                "Chat error:",
                error
            );

            state.aiContent.innerHTML = `
                <div class="error-message">
                    ${escapeHTML(
                error.message ||
                "Something went wrong."
            )}
                </div>
            `;

            if (statusText) {

                statusText.textContent =
                    "● Backend Error";
            }
        }

    } finally {

        isGenerating = false;

        currentAbortController =
            null;

        if (sendBtn) {
            sendBtn.disabled = false;
        }

        if (stopBtn) {
            stopBtn.classList.add("hidden");
        }

        if (userInput) {
            userInput.focus();
        }

        scrollToBottom();
    }
}

// ============================================================
// STOP GENERATION
// ============================================================

function stopGenerating() {

    if (!isGenerating) {
        return;
    }

    if (currentAbortController) {

        currentAbortController.abort();

        currentAbortController =
            null;
    }

    isGenerating = false;

    if (sendBtn) {
        sendBtn.disabled = false;
    }

    if (stopBtn) {
        stopBtn.classList.add("hidden");
    }

    if (statusText) {

        statusText.textContent =
            "● Generation stopped";
    }

    if (userInput) {
        userInput.focus();
    }
}

// ============================================================
// REGENERATE
// ============================================================

async function regenerateResponse(
    messageDiv
) {

    if (isGenerating) {
        return;
    }

    if (!currentSessionId) {
        return;
    }

    if (!messageDiv) {
        return;
    }

    isGenerating = true;

    currentAbortController =
        new AbortController();

    if (sendBtn) {
        sendBtn.disabled = true;
    }

    if (stopBtn) {
        stopBtn.classList.remove("hidden");
    }

    if (statusText) {

        statusText.textContent =
            "● Regenerating...";
    }

    const aiContent =
        messageDiv.querySelector(
            ".message-content"
        );

    if (!aiContent) {

        isGenerating = false;

        return;
    }

    const actions =
        messageDiv.querySelector(
            ".message-actions"
        );

    if (actions) {
        actions.remove();
    }

    aiContent.innerHTML = `
        <span class="loading-dots">
            <span></span>
            <span></span>
            <span></span>
        </span>
    `;

    const state = {

        aiContent,

        messageDiv,

        fullResponse:
            "",

        completed:
            false,

        streamError:
            null
    };

    try {

        const response =
            await fetch(
                `${API_URL}/chats/${encodeURIComponent(currentSessionId)}/regenerate`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    signal:
                        currentAbortController.signal
                }
            );

        if (!response.ok) {

            let errorMessage =
                `Server error: ${response.status}`;

            try {

                const errorData =
                    await response.json();

                errorMessage =
                    errorData.detail ||
                    errorData.message ||
                    errorMessage;

            } catch (_) { }

            throw new Error(
                errorMessage
            );
        }

        await readStreamingResponse(
            response,
            state
        );

        if (
            !state.completed &&
            state.fullResponse
        ) {

            aiContent.innerHTML =
                formatMessage(
                    state.fullResponse
                );

            showRegenerateButton(
                messageDiv
            );
        }

        await loadChatHistoryList();

    } catch (error) {

        if (
            error.name ===
            "AbortError"
        ) {

            if (statusText) {

                statusText.textContent =
                    "● Generation stopped";
            }

            aiContent.innerHTML =
                formatMessage(
                    state.fullResponse
                );

        } else {

            console.error(
                "Regeneration error:",
                error
            );

            aiContent.innerHTML = `
                <div class="error-message">
                    ${escapeHTML(
                error.message ||
                "Regeneration failed."
            )}
                </div>
            `;

            if (statusText) {

                statusText.textContent =
                    "● Backend Error";
            }
        }

    } finally {

        isGenerating = false;

        currentAbortController =
            null;

        if (sendBtn) {
            sendBtn.disabled = false;
        }

        if (stopBtn) {
            stopBtn.classList.add("hidden");
        }

        if (userInput) {
            userInput.focus();
        }

        scrollToBottom();
    }
}

// ============================================================
// CREATE NEW CHAT
// ============================================================

async function createNewChat() {

    if (isGenerating) {
        return null;
    }

    try {

        console.log(
            "Creating new chat..."
        );

        const response =
            await fetch(
                `${API_URL}/chats`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        title: "New Chat"
                    })
                }
            );

        console.log(
            "Create chat status:",
            response.status
        );

        const data =
            await response.json();

        console.log(
            "Create chat response:",
            data
        );

        if (!response.ok) {

            throw new Error(
                data.detail ||
                data.message ||
                `Unable to create chat: ${response.status}`
            );
        }

        const chat =
            data.chat;

        if (
            !data.success ||
            !chat ||
            !chat.id
        ) {

            throw new Error(
                "Invalid chat response from server."
            );
        }

        // UUID MUST remain string
        currentSessionId =
            String(chat.id);

        localStorage.setItem(
            "currentSessionId",
            currentSessionId
        );

        console.log(
            "Current session ID:",
            currentSessionId
        );

        showWelcome();

        await loadChatHistoryList();

        if (statusText) {

            statusText.textContent =
                "● Online";
        }

        return chat;

    } catch (error) {

        console.error(
            "Create chat error:",
            error
        );

        if (statusText) {

            statusText.textContent =
                "● Backend Error";
        }

        alert(
            error.message ||
            "Unable to create a new chat."
        );

        return null;
    }
}

// ============================================================
// LOAD CHAT HISTORY
// ============================================================

async function loadChatHistoryList() {

    if (!chatHistory) {
        return [];
    }

    try {

        const response =
            await fetch(
                `${API_URL}/chats`
            );

        if (!response.ok) {

            throw new Error(
                "Unable to load chats."
            );
        }

        const data =
            await response.json();

        const chats =
            Array.isArray(data.chats)
                ? data.chats
                : [];

        // ----------------------------------------------------
        // IMPORTANT FIX
        //
        // If saved session doesn't exist anymore,
        // automatically clear it.
        // ----------------------------------------------------

        if (currentSessionId) {

            const sessionExists =
                chats.some(function (chat) {

                    return (
                        String(chat.id) ===
                        String(currentSessionId)
                    );

                });

            if (!sessionExists) {

                console.warn(
                    "Saved session no longer exists:",
                    currentSessionId
                );

                currentSessionId =
                    null;

                localStorage.removeItem(
                    "currentSessionId"
                );
            }
        }

        chatHistory.innerHTML = "";

        if (!chats.length) {

            chatHistory.innerHTML = `
                <div class="empty-history">
                    No chats yet
                </div>
            `;

            return chats;
        }

        chats.forEach(function (chat) {

            const item =
                document.createElement("div");

            item.className =
                "chat-history-item";

            if (
                String(chat.id) ===
                String(currentSessionId)
            ) {

                item.classList.add(
                    "active"
                );
            }

            // ------------------------------------------------
            // MAIN
            // ------------------------------------------------

            const main =
                document.createElement("div");

            main.className =
                "chat-history-main";

            const title =
                document.createElement("span");

            title.className =
                "chat-history-title";

            title.textContent =
                chat.title ||
                "New Chat";

            main.appendChild(title);

            main.addEventListener(
                "click",
                async function () {

                    if (isGenerating) {
                        return;
                    }

                    await loadSession(
                        String(chat.id)
                    );
                }
            );

            // ------------------------------------------------
            // ACTIONS
            // ------------------------------------------------

            const actions =
                document.createElement("div");

            actions.className =
                "chat-history-actions";

            // ------------------------------------------------
            // Rename
            // ------------------------------------------------

            const renameBtn =
                document.createElement("button");

            renameBtn.className =
                "rename-chat-btn";

            renameBtn.type =
                "button";

            renameBtn.title =
                "Rename chat";

            renameBtn.textContent =
                "✏️";

            renameBtn.addEventListener(
                "click",
                async function (event) {

                    event.stopPropagation();

                    await renameChat(
                        String(chat.id),
                        chat.title
                    );
                }
            );

            // ------------------------------------------------
            // Delete
            // ------------------------------------------------

            const deleteBtn =
                document.createElement("button");

            deleteBtn.className =
                "delete-chat-btn";

            deleteBtn.type =
                "button";

            deleteBtn.title =
                "Delete chat";

            deleteBtn.textContent =
                "🗑️";

            deleteBtn.addEventListener(
                "click",
                async function (event) {

                    event.stopPropagation();

                    await deleteChat(
                        String(chat.id)
                    );
                }
            );

            actions.appendChild(
                renameBtn
            );

            actions.appendChild(
                deleteBtn
            );

            item.appendChild(main);
            item.appendChild(actions);

            chatHistory.appendChild(item);
        });

        return chats;

    } catch (error) {

        console.error(
            "Chat history error:",
            error
        );

        chatHistory.innerHTML = `
            <div class="empty-history">
                Unable to load chat history
            </div>
        `;

        return [];
    }
}

// ============================================================
// LOAD SESSION
// ============================================================

async function loadSession(
    sessionId
) {

    if (!sessionId) {
        return false;
    }

    try {

        const response =
            await fetch(
                `${API_URL}/chats/${encodeURIComponent(sessionId)}`
            );

        // ----------------------------------------------------
        // IMPORTANT 404 FIX
        // ----------------------------------------------------

        if (response.status === 404) {

            console.warn(
                "Chat not found. Removing stale session:",
                sessionId
            );

            if (
                String(currentSessionId) ===
                String(sessionId)
            ) {

                currentSessionId =
                    null;

                localStorage.removeItem(
                    "currentSessionId"
                );
            }

            // Create replacement chat
            const newChat =
                await createNewChat();

            return !!newChat;
        }

        if (!response.ok) {

            let errorMessage =
                "Unable to load chat.";

            try {

                const errorData =
                    await response.json();

                errorMessage =
                    errorData.detail ||
                    errorData.message ||
                    errorMessage;

            } catch (_) { }

            throw new Error(
                errorMessage
            );
        }

        const data =
            await response.json();

        currentSessionId =
            String(
                data.chat?.id ||
                sessionId
            );

        localStorage.setItem(
            "currentSessionId",
            currentSessionId
        );

        if (chatMessages) {
            chatMessages.innerHTML = "";
        }

        const messages =
            Array.isArray(data.messages)
                ? data.messages
                : [];

        if (!messages.length) {

            showWelcome();

        } else {

            messages.forEach(
                function (message) {

                    const role =
                        message.role === "user"
                            ? "user"
                            : "assistant";

                    addMessage(
                        message.content || "",
                        role,
                        false
                    );
                }
            );

            const assistantMessages =
                chatMessages
                    ? chatMessages.querySelectorAll(
                        ".assistant-message"
                    )
                    : [];

            if (assistantMessages.length) {

                const latest =
                    assistantMessages[
                    assistantMessages.length - 1
                    ];

                showRegenerateButton(
                    latest
                );
            }
        }

        await loadChatHistoryList();

        if (statusText) {

            statusText.textContent =
                "● Online";
        }

        scrollToBottom();

        return true;

    } catch (error) {

        console.error(
            "Load session error:",
            error
        );

        // ----------------------------------------------------
        // If old session fails, clear it and create new one.
        // ----------------------------------------------------

        if (
            String(currentSessionId) ===
            String(sessionId)
        ) {

            currentSessionId =
                null;

            localStorage.removeItem(
                "currentSessionId"
            );
        }

        const newChat =
            await createNewChat();

        return !!newChat;
    }
}

// ============================================================
// RENAME CHAT
// ============================================================

async function renameChat(
    sessionId,
    oldTitle
) {

    const title =
        prompt(
            "Enter new chat title:",
            oldTitle || "New Chat"
        );

    if (title === null) {
        return;
    }

    const cleanTitle =
        title.trim();

    if (!cleanTitle) {

        alert(
            "Chat title cannot be empty."
        );

        return;
    }

    try {

        const response =
            await fetch(
                `${API_URL}/chats/${encodeURIComponent(sessionId)}`,
                {
                    method: "PUT",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        title: cleanTitle
                    })
                }
            );

        if (!response.ok) {

            const data =
                await response.json()
                    .catch(() => ({}));

            throw new Error(
                data.detail ||
                data.message ||
                "Unable to rename chat."
            );
        }

        await loadChatHistoryList();

    } catch (error) {

        console.error(
            "Rename error:",
            error
        );

        alert(
            error.message ||
            "Unable to rename chat."
        );
    }
}

// ============================================================
// DELETE CHAT
// ============================================================

async function deleteChat(
    sessionId
) {

    const confirmed =
        confirm(
            "Delete this chat permanently?"
        );

    if (!confirmed) {
        return;
    }

    try {

        const response =
            await fetch(
                `${API_URL}/chats/${encodeURIComponent(sessionId)}`,
                {
                    method: "DELETE"
                }
            );

        if (!response.ok) {

            const data =
                await response.json()
                    .catch(() => ({}));

            throw new Error(
                data.detail ||
                data.message ||
                "Unable to delete chat."
            );
        }

        // ----------------------------------------------------
        // If currently selected chat was deleted
        // ----------------------------------------------------

        if (
            String(sessionId) ===
            String(currentSessionId)
        ) {

            currentSessionId =
                null;

            localStorage.removeItem(
                "currentSessionId"
            );

            showWelcome();

            await createNewChat();

        } else {

            await loadChatHistoryList();
        }

    } catch (error) {

        console.error(
            "Delete chat error:",
            error
        );

        alert(
            error.message ||
            "Unable to delete chat."
        );
    }
}

// ============================================================
// CLEAR CHAT UI
// ============================================================

function clearChatUI() {

    if (!chatMessages) {
        return;
    }

    chatMessages.innerHTML = "";

    showWelcome();
}

// ============================================================
// MEMORY
// ============================================================

async function loadMemory() {

    if (!memoryContent) {
        return;
    }

    memoryContent.innerHTML = `
        <div class="loading-memory">
            Loading memory...
        </div>
    `;

    try {

        const response =
            await fetch(
                `${API_URL}/memory?query=user%20assistant%20conversation`
            );

        if (!response.ok) {

            throw new Error(
                "Unable to load memory."
            );
        }

        const data =
            await response.json();

        const memories =
            Array.isArray(data.memories)
                ? data.memories
                : [];

        if (!memories.length) {

            memoryContent.innerHTML = `
                <div class="empty-memory">
                    No memories saved yet.
                </div>
            `;

            return;
        }

        memoryContent.innerHTML = "";

        memories.forEach(
            function (memory) {

                const item =
                    document.createElement("div");

                item.className =
                    "memory-item";

                item.innerHTML = `
                    <div class="memory-type">
                        ${escapeHTML(
                    memory.memory_type ||
                    "Memory"
                )}
                    </div>

                    <div class="memory-text">
                        ${escapeHTML(
                    memory.content ||
                    ""
                )}
                    </div>

                    <div class="memory-date">
                        ${escapeHTML(
                    memory.created_at ||
                    ""
                )}
                    </div>
                `;

                memoryContent.appendChild(
                    item
                );
            }
        );

    } catch (error) {

        console.error(
            "Memory error:",
            error
        );

        memoryContent.innerHTML = `
            <div class="error-message">
                Unable to load memory.
            </div>
        `;
    }
}

// ============================================================
// LOAD DOCUMENTS
// ============================================================

async function loadDocuments() {

    if (!documentList) {
        return;
    }

    try {

        const response =
            await fetch(
                `${API_URL}/documents`
            );

        if (!response.ok) {

            throw new Error(
                "Unable to load documents."
            );
        }

        const data =
            await response.json();

        const documents =
            Array.isArray(data.documents)
                ? data.documents
                : [];

        documentList.innerHTML = "";

        if (!documents.length) {

            documentList.innerHTML = `
                <div class="empty-documents">
                    No documents uploaded.
                </div>
            `;

            return;
        }

        documents.forEach(
            function (doc) {

                const item =
                    document.createElement("div");

                item.className =
                    "document-item";

                const filename =
                    document.createElement("span");

                filename.className =
                    "document-name";

                filename.textContent =
                    doc.filename ||
                    "Unknown document";

                const deleteBtn =
                    document.createElement("button");

                deleteBtn.className =
                    "delete-document-btn";

                deleteBtn.type =
                    "button";

                deleteBtn.textContent =
                    "🗑️";

                deleteBtn.addEventListener(
                    "click",
                    async function () {

                        await deleteDocument(
                            doc.filename
                        );
                    }
                );

                item.appendChild(
                    filename
                );

                item.appendChild(
                    deleteBtn
                );

                documentList.appendChild(
                    item
                );
            }
        );

    } catch (error) {

        console.error(
            "Documents error:",
            error
        );

        documentList.innerHTML = `
            <div class="error-message">
                Unable to load documents.
            </div>
        `;
    }
}

// ============================================================
// UPLOAD PDF
// ============================================================

async function uploadPDF() {

    if (!pdfInput) {
        return;
    }

    const file =
        pdfInput.files[0];

    if (!file) {

        alert(
            "Please select a PDF file."
        );

        return;
    }

    if (
        !file.name
            .toLowerCase()
            .endsWith(".pdf")
    ) {

        alert(
            "Only PDF files are allowed."
        );

        return;
    }

    const formData =
        new FormData();

    formData.append(
        "file",
        file
    );

    if (uploadBtn) {
        uploadBtn.disabled = true;
    }

    if (uploadStatus) {

        uploadStatus.textContent =
            "Uploading and indexing...";
    }

    try {

        const response =
            await fetch(
                `${API_URL}/documents/upload`,
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail ||
                data.message ||
                "Upload failed."
            );
        }

        if (uploadStatus) {

            uploadStatus.textContent =
                `✓ ${file.name} uploaded and indexed`;
        }

        pdfInput.value = "";

        await loadDocuments();

    } catch (error) {

        console.error(
            "Upload error:",
            error
        );

        if (uploadStatus) {

            uploadStatus.textContent =
                `✗ ${error.message}`;
        }

    } finally {

        if (uploadBtn) {
            uploadBtn.disabled = false;
        }
    }
}

// ============================================================
// DELETE DOCUMENT
// ============================================================

async function deleteDocument(
    filename
) {

    const confirmed =
        confirm(
            `Delete "${filename}"?`
        );

    if (!confirmed) {
        return;
    }

    try {

        const response =
            await fetch(
                `${API_URL}/documents/${encodeURIComponent(filename)}`,
                {
                    method: "DELETE"
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail ||
                data.message ||
                "Delete failed."
            );
        }

        await loadDocuments();

    } catch (error) {

        console.error(
            "Delete document error:",
            error
        );

        alert(
            error.message ||
            "Unable to delete document."
        );
    }
}

// ============================================================
// SUGGESTIONS
// ============================================================

function attachSuggestionEvents() {

    document
        .querySelectorAll(".suggestion-btn")
        .forEach(function (button) {

            // Prevent duplicate listeners
            if (
                button.dataset.listenerAttached ===
                "true"
            ) {
                return;
            }

            button.dataset.listenerAttached =
                "true";

            button.addEventListener(
                "click",
                function () {

                    if (!userInput) {
                        return;
                    }

                    userInput.value =
                        button.textContent.trim();

                    autoResize();

                    userInput.focus();
                }
            );
        });
}

// ============================================================
// AUTO RESIZE
// ============================================================

function autoResize() {

    if (!userInput) {
        return;
    }

    userInput.style.height =
        "auto";

    userInput.style.height =
        Math.min(
            userInput.scrollHeight,
            180
        ) + "px";
}

// ============================================================
// EVENT LISTENERS
// ============================================================

// SEND
if (sendBtn) {

    sendBtn.addEventListener(
        "click",
        sendMessage
    );
}

// STOP
if (stopBtn) {

    stopBtn.addEventListener(
        "click",
        stopGenerating
    );
}

// ENTER
if (userInput) {

    userInput.addEventListener(
        "keydown",
        function (event) {

            if (
                event.key === "Enter" &&
                !event.shiftKey
            ) {

                event.preventDefault();

                sendMessage();
            }
        }
    );

    userInput.addEventListener(
        "input",
        autoResize
    );
}

// NEW CHAT
if (newChatBtn) {

    newChatBtn.addEventListener(
        "click",
        async function () {

            await createNewChat();

        }
    );
}

// CLEAR
if (clearBtn) {

    clearBtn.addEventListener(
        "click",
        function () {

            if (isGenerating) {
                return;
            }

            clearChatUI();

        }
    );
}

// MEMORY OPEN
if (memoryBtn) {

    memoryBtn.addEventListener(
        "click",
        async function () {

            if (memoryPanel) {

                memoryPanel.classList.add(
                    "open"
                );

                memoryPanel.classList.remove(
                    "hidden"
                );
            }

            await loadMemory();

        }
    );
}

// MEMORY CLOSE
if (closeMemoryBtn) {

    closeMemoryBtn.addEventListener(
        "click",
        function () {

            if (memoryPanel) {

                memoryPanel.classList.remove(
                    "open"
                );

                memoryPanel.classList.add(
                    "hidden"
                );
            }
        }
    );
}

// UPLOAD
if (uploadBtn) {

    uploadBtn.addEventListener(
        "click",
        uploadPDF
    );
}

// ============================================================
// BACKEND HEALTH
// ============================================================

async function checkBackendHealth() {

    try {

        const response =
            await fetch(
                `${API_URL}/health`
            );

        if (!response.ok) {

            throw new Error(
                "Backend unavailable"
            );
        }

        const data =
            await response.json();

        console.log(
            "Backend:",
            data
        );

        if (statusText) {

            statusText.textContent =
                "● Online";
        }

        return true;

    } catch (error) {

        console.error(
            "Backend health error:",
            error
        );

        if (statusText) {

            statusText.textContent =
                "● Backend Offline";
        }

        return false;
    }
}

// ============================================================
// INITIALIZE APP
// ============================================================

async function initializeApp() {

    console.log(
        "🚀 Initializing AI Chatbot..."
    );

    const backendOnline =
        await checkBackendHealth();

    if (!backendOnline) {
        return;
    }

    try {

        // ----------------------------------------------------
        // Load chats
        // ----------------------------------------------------

        const chats =
            await loadChatHistoryList();

        // ----------------------------------------------------
        // Load saved session
        // ----------------------------------------------------

        if (currentSessionId) {

            console.log(
                "Loading saved session:",
                currentSessionId
            );

            const sessionLoaded =
                await loadSession(
                    currentSessionId
                );

            if (!sessionLoaded) {

                currentSessionId =
                    null;

                localStorage.removeItem(
                    "currentSessionId"
                );
            }
        }

        // ----------------------------------------------------
        // IMPORTANT
        //
        // If there was no saved chat or old chat was deleted,
        // create a new chat automatically.
        // ----------------------------------------------------

        if (!currentSessionId) {

            console.log(
                "No valid session found. Creating new chat..."
            );

            await createNewChat();
        }

        // ----------------------------------------------------
        // Documents
        // ----------------------------------------------------

        await loadDocuments();

        // ----------------------------------------------------
        // Suggestions
        // ----------------------------------------------------

        attachSuggestionEvents();

        // ----------------------------------------------------
        // Input
        // ----------------------------------------------------

        autoResize();

        console.log(
            "✅ AI Chatbot frontend connected successfully"
        );

    } catch (error) {

        console.error(
            "Initialization error:",
            error
        );

        if (statusText) {

            statusText.textContent =
                "● Frontend Error";
        }
    }
}

// ============================================================
// START
// ============================================================

initializeApp();

// ============================================================
// GLOBAL FUNCTIONS
// ============================================================

window.sendMessage =
    sendMessage;

window.stopGenerating =
    stopGenerating;

window.createNewChat =
    createNewChat;

window.copyCode =
    copyCode;

window.uploadPDF =
    uploadPDF;

window.loadMemory =
    loadMemory;

window.loadDocuments =
    loadDocuments;

window.loadSession =
    loadSession;

window.regenerateResponse =
    regenerateResponse;

window.renameChat =
    renameChat;

window.deleteChat =
    deleteChat;