// ============================================================
// AI MEMORY + RAG CHATBOT
// COMPLETE FRONTEND SCRIPT
// ============================================================

const API_URL = "http://127.0.0.1:8000";


// ============================================================
// GLOBAL STATE
// ============================================================

let currentSessionId = localStorage.getItem("currentSessionId")
    ? Number(localStorage.getItem("currentSessionId"))
    : null;

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
// CLEAN DEEPSEEK THINKING
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

    // Unclosed <think> block
    cleaned = cleaned.replace(
        /<think>[\s\S]*$/gi,
        ""
    );

    // Any remaining tags
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

            let lang = language || "text";

            code = code.replace(/\n$/, "");

            codeBlocks.push({
                language: lang,
                code: code
            });

            return `@@CODEBLOCK_${index}@@`;
        }
    );

    // --------------------------------------------------------
    // Escape normal HTML
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
        /(?<!\*)\*([^*\n]+)\*(?!\*)/g,
        "<em>$1</em>"
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
        /(?:^|\n)(?:[-*]) (.+)(?=\n|$)/g,
        function (_, item) {

            return `\n<li>${item}</li>`;
        }
    );

    source = source.replace(
        /((?:<li>.*<\/li>\n?)+)/g,
        function (list) {

            return `<ul>${list}</ul>`;
        }
    );

    // --------------------------------------------------------
    // Ordered lists
    // --------------------------------------------------------

    source = source.replace(
        /(?:^|\n)\d+\.\s+(.+)(?=\n|$)/g,
        function (_, item) {

            return `\n<oli>${item}</oli>`;
        }
    );

    source = source.replace(
        /((?:<oli>.*<\/oli>\n?)+)/g,
        function (list) {

            return `<ol>${list.replace(
                /<oli>/g,
                "<li>"
            ).replace(
                /<\/oli>/g,
                "</li>"
            )}</ol>`;
        }
    );

    // --------------------------------------------------------
    // Horizontal line
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

        const escapedCode = escapeHTML(
            block.code
        );

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

    if (
        typeof hljs === "undefined"
    ) {
        return;
    }

    document
        .querySelectorAll(
            ".code-wrapper pre code"
        )
        .forEach(function (block) {

            try {

                if (
                    !block.dataset.highlighted
                ) {

                    hljs.highlightElement(
                        block
                    );
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
            button.getAttribute(
                "data-code"
            );

        const code = decodeURIComponent(
            encodedCode || ""
        );

        await navigator.clipboard.writeText(
            code
        );

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
// SCROLL CHAT TO BOTTOM
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

    contentWrapper.appendChild(
        content
    );

    // --------------------------------------------------------
    // Assistant action buttons
    // --------------------------------------------------------

    if (
        role === "assistant" &&
        showActions
    ) {

        const actions =
            document.createElement("div");

        actions.className =
            "message-actions";

        const regenerateBtn =
            document.createElement("button");

        regenerateBtn.className =
            "regenerate-btn";

        regenerateBtn.type =
            "button";

        regenerateBtn.innerHTML =
            "↻ Regenerate";

        regenerateBtn.addEventListener(
            "click",
            function () {

                regenerateResponse(
                    messageDiv
                );

            }
        );

        actions.appendChild(
            regenerateBtn
        );

        contentWrapper.appendChild(
            actions
        );
    }

    messageDiv.appendChild(
        avatar
    );

    messageDiv.appendChild(
        contentWrapper
    );

    chatMessages.appendChild(
        messageDiv
    );

    applySyntaxHighlighting();

    scrollToBottom();

    return messageDiv;
}


// ============================================================
// CREATE STREAMING MESSAGE
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

    const contentWrapper =
        document.createElement("div");

    contentWrapper.className =
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

    contentWrapper.appendChild(
        content
    );

    messageDiv.appendChild(
        avatar
    );

    messageDiv.appendChild(
        contentWrapper
    );

    chatMessages.appendChild(
        messageDiv
    );

    scrollToBottom();

    return {
        messageDiv,
        content
    };
}


// ============================================================
// SHOW REGENERATE BUTTON
// ============================================================

function showRegenerateButton(
    messageDiv
) {

    if (!messageDiv) {
        return;
    }

    // Remove old action row
    const oldActions =
        messageDiv.querySelector(
            ".message-actions"
        );

    if (oldActions) {
        oldActions.remove();
    }

    const wrapper =
        messageDiv.querySelector(
            ".message-content-wrapper"
        );

    if (!wrapper) {
        return;
    }

    const actions =
        document.createElement("div");

    actions.className =
        "message-actions";

    const button =
        document.createElement("button");

    button.type =
        "button";

    button.className =
        "regenerate-btn";

    button.innerHTML =
        "↻ Regenerate";

    button.addEventListener(
        "click",
        function () {

            regenerateResponse(
                messageDiv
            );

        }
    );

    actions.appendChild(
        button
    );

    wrapper.appendChild(
        actions
    );
}


// ============================================================
// HIDE ALL REGENERATE BUTTONS
// ============================================================

function hideAllRegenerateButtons() {

    document
        .querySelectorAll(
            ".message-actions"
        )
        .forEach(function (element) {

            element.remove();

        });
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

        data = JSON.parse(
            line
        );

    } catch (error) {

        console.warn(
            "Invalid stream JSON:",
            line
        );

        return;
    }

    // --------------------------------------------------------
    // STREAM CHUNK
    // --------------------------------------------------------

    if (
        data.type === "chunk"
    ) {

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
    // STREAM COMPLETE
    // --------------------------------------------------------

    if (
        data.type === "done"
    ) {

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

        if (
            data.chat &&
            data.chat.id
        ) {

            currentSessionId =
                Number(data.chat.id);

            localStorage.setItem(
                "currentSessionId",
                currentSessionId
            );
        }

        statusText.textContent =
            "● Online";

        scrollToBottom();

        return;
    }

    // --------------------------------------------------------
    // ERROR
    // --------------------------------------------------------

    if (
        data.type === "error"
    ) {

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
// SEND MESSAGE
// ============================================================

async function sendMessage() {

    if (isGenerating) {
        return;
    }

    const message =
        userInput.value.trim();

    if (!message) {
        return;
    }

    // --------------------------------------------------------
    // Create session if required
    // --------------------------------------------------------

    if (!currentSessionId) {

        await createNewChat();
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
        chatMessages.querySelector(
            ".welcome-screen"
        );

    if (welcome) {
        welcome.remove();
    }

    // --------------------------------------------------------
    // Hide previous regenerate buttons
    // --------------------------------------------------------

    hideAllRegenerateButtons();

    // --------------------------------------------------------
    // Show user message
    // --------------------------------------------------------

    addMessage(
        message,
        "user"
    );

    userInput.value = "";

    autoResize();

    // --------------------------------------------------------
    // Generation state
    // --------------------------------------------------------

    isGenerating = true;

    currentAbortController =
        new AbortController();

    sendBtn.disabled = true;

    if (stopBtn) {

        stopBtn.classList.remove(
            "hidden"
        );
    }

    statusText.textContent =
        "● Thinking...";

    // --------------------------------------------------------
    // Create AI streaming message
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

        fullResponse: "",

        completed: false,

        streamError: null

    };

    try {

        // ----------------------------------------------------
        // API REQUEST
        // ----------------------------------------------------

        const response =
            await fetch(
                `${API_URL}/chats/${currentSessionId}/messages`,
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
        // HTTP ERROR
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

        if (!response.body) {

            throw new Error(
                "Streaming response body is unavailable."
            );
        }

        // ----------------------------------------------------
        // READ STREAM
        // ----------------------------------------------------

        const reader =
            response.body.getReader();

        const decoder =
            new TextDecoder(
                "utf-8"
            );

        let buffer = "";

        while (true) {

            const result =
                await reader.read();

            if (result.done) {
                break;
            }

            buffer += decoder.decode(
                result.value,
                {
                    stream: true
                }
            );

            const lines =
                buffer.split("\n");

            buffer =
                lines.pop() || "";

            for (
                const line of lines
            ) {

                processStreamLine(
                    line,
                    state
                );
            }
        }

        // ----------------------------------------------------
        // PROCESS LAST LINE
        // ----------------------------------------------------

        if (buffer.trim()) {

            processStreamLine(
                buffer,
                state
            );
        }

        // ----------------------------------------------------
        // FALLBACK
        // ----------------------------------------------------

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

        // ----------------------------------------------------
        // REFRESH SIDEBAR
        // ----------------------------------------------------

        await loadChatHistoryList();

    } catch (error) {

        // ----------------------------------------------------
        // USER STOPPED GENERATION
        // ----------------------------------------------------

        if (
            error.name ===
            "AbortError"
        ) {

            statusText.textContent =
                "● Generation stopped";

            state.aiContent.innerHTML =
                formatMessage(
                    state.fullResponse
                );

            if (
                state.fullResponse
            ) {

                const stoppedNotice =
                    document.createElement(
                        "div"
                    );

                stoppedNotice.className =
                    "generation-stopped";

                stoppedNotice.textContent =
                    "Generation stopped.";

                state.aiContent.appendChild(
                    stoppedNotice
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

            statusText.textContent =
                "● Backend Error";
        }

    } finally {

        isGenerating = false;

        currentAbortController =
            null;

        sendBtn.disabled = false;

        if (stopBtn) {

            stopBtn.classList.add(
                "hidden"
            );
        }

        userInput.focus();

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

    if (
        currentAbortController
    ) {

        currentAbortController.abort();

        currentAbortController =
            null;
    }

    isGenerating = false;

    sendBtn.disabled = false;

    if (stopBtn) {

        stopBtn.classList.add(
            "hidden"
        );
    }

    statusText.textContent =
        "● Generation stopped";

    userInput.focus();
}


// ============================================================
// REGENERATE LAST RESPONSE
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

    // --------------------------------------------------------
    // Generation state
    // --------------------------------------------------------

    isGenerating = true;

    currentAbortController =
        new AbortController();

    sendBtn.disabled = true;

    if (stopBtn) {

        stopBtn.classList.remove(
            "hidden"
        );
    }

    statusText.textContent =
        "● Regenerating...";

    // --------------------------------------------------------
    // Find AI content
    // --------------------------------------------------------

    const aiContent =
        messageDiv.querySelector(
            ".message-content"
        );

    if (!aiContent) {

        isGenerating = false;

        return;
    }

    // --------------------------------------------------------
    // Remove actions
    // --------------------------------------------------------

    const actions =
        messageDiv.querySelector(
            ".message-actions"
        );

    if (actions) {
        actions.remove();
    }

    // --------------------------------------------------------
    // Reset response
    // --------------------------------------------------------

    aiContent.innerHTML = `
        <span class="loading-dots">
            <span></span>
            <span></span>
            <span></span>
        </span>
    `;

    const state = {

        aiContent:
            aiContent,

        messageDiv:
            messageDiv,

        fullResponse: "",

        completed: false,

        streamError: null

    };

    try {

        // ----------------------------------------------------
        // REGENERATE REQUEST
        // ----------------------------------------------------

        const response =
            await fetch(
                `${API_URL}/chats/${currentSessionId}/regenerate`,
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

        if (!response.body) {

            throw new Error(
                "Regeneration stream unavailable."
            );
        }

        // ----------------------------------------------------
        // READ STREAM
        // ----------------------------------------------------

        const reader =
            response.body.getReader();

        const decoder =
            new TextDecoder(
                "utf-8"
            );

        let buffer = "";

        while (true) {

            const result =
                await reader.read();

            if (result.done) {
                break;
            }

            buffer += decoder.decode(
                result.value,
                {
                    stream: true
                }
            );

            const lines =
                buffer.split("\n");

            buffer =
                lines.pop() || "";

            for (
                const line of lines
            ) {

                processStreamLine(
                    line,
                    state
                );
            }
        }

        // ----------------------------------------------------
        // LAST LINE
        // ----------------------------------------------------

        if (buffer.trim()) {

            processStreamLine(
                buffer,
                state
            );
        }

        await loadChatHistoryList();

    } catch (error) {

        if (
            error.name ===
            "AbortError"
        ) {

            statusText.textContent =
                "● Generation stopped";

            aiContent.innerHTML =
                formatMessage(
                    state.fullResponse
                );

            if (state.fullResponse) {

                const stoppedNotice =
                    document.createElement(
                        "div"
                    );

                stoppedNotice.className =
                    "generation-stopped";

                stoppedNotice.textContent =
                    "Generation stopped.";

                aiContent.appendChild(
                    stoppedNotice
                );
            }

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

            statusText.textContent =
                "● Backend Error";
        }

    } finally {

        isGenerating = false;

        currentAbortController =
            null;

        sendBtn.disabled = false;

        if (stopBtn) {

            stopBtn.classList.add(
                "hidden"
            );
        }

        userInput.focus();

        scrollToBottom();
    }
}


// ============================================================
// CREATE NEW CHAT
// ============================================================

async function createNewChat() {

    if (isGenerating) {
        return;
    }

    try {

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

        if (!response.ok) {

            throw new Error(
                `Unable to create chat: ${response.status}`
            );
        }

        const data =
            await response.json();

        const chat =
            data.chat;

        if (
            !chat ||
            !chat.id
        ) {

            throw new Error(
                "Invalid chat response."
            );
        }

        currentSessionId =
            Number(chat.id);

        localStorage.setItem(
            "currentSessionId",
            currentSessionId
        );

        showWelcome();

        await loadChatHistoryList();

        statusText.textContent =
            "● Online";

    } catch (error) {

        console.error(
            "Create chat error:",
            error
        );

        alert(
            "Unable to create a new chat."
        );
    }
}


// ============================================================
// LOAD CHAT HISTORY LIST
// ============================================================

async function loadChatHistoryList() {

    if (!chatHistory) {
        return;
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
            data.chats || [];

        chatHistory.innerHTML = "";

        if (!chats.length) {

            chatHistory.innerHTML = `
                <div class="empty-history">
                    No chats yet
                </div>
            `;

            return;
        }

        chats.forEach(
            function (chat) {

                const item =
                    document.createElement(
                        "div"
                    );

                item.className =
                    "chat-history-item";

                if (
                    Number(chat.id) ===
                    Number(currentSessionId)
                ) {

                    item.classList.add(
                        "active"
                    );
                }

                const main =
                    document.createElement(
                        "div"
                    );

                main.className =
                    "chat-history-main";

                const title =
                    document.createElement(
                        "span"
                    );

                title.className =
                    "chat-history-title";

                title.textContent =
                    chat.title ||
                    "New Chat";

                main.appendChild(
                    title
                );

                main.addEventListener(
                    "click",
                    async function () {

                        if (isGenerating) {
                            return;
                        }

                        await loadSession(
                            Number(chat.id)
                        );

                    }
                );

                const actions =
                    document.createElement(
                        "div"
                    );

                actions.className =
                    "chat-history-actions";

                // ------------------------------------------------
                // Rename
                // ------------------------------------------------

                const renameBtn =
                    document.createElement(
                        "button"
                    );

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
                            Number(chat.id),
                            chat.title
                        );

                    }
                );

                // ------------------------------------------------
                // Delete
                // ------------------------------------------------

                const deleteBtn =
                    document.createElement(
                        "button"
                    );

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
                            Number(chat.id)
                        );

                    }
                );

                actions.appendChild(
                    renameBtn
                );

                actions.appendChild(
                    deleteBtn
                );

                item.appendChild(
                    main
                );

                item.appendChild(
                    actions
                );

                chatHistory.appendChild(
                    item
                );

            }
        );

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
    }
}


// ============================================================
// LOAD SINGLE SESSION
// ============================================================

async function loadSession(
    sessionId
) {

    try {

        const response =
            await fetch(
                `${API_URL}/chats/${sessionId}`
            );

        if (!response.ok) {

            throw new Error(
                "Unable to load chat."
            );
        }

        const data =
            await response.json();

        currentSessionId =
            Number(sessionId);

        localStorage.setItem(
            "currentSessionId",
            currentSessionId
        );

        chatMessages.innerHTML = "";

        const messages =
            data.messages || [];

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

            // Only latest AI response gets regenerate
            const assistantMessages =
                chatMessages.querySelectorAll(
                    ".assistant-message"
                );

            if (
                assistantMessages.length
            ) {

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

        statusText.textContent =
            "● Online";

        scrollToBottom();

    } catch (error) {

        console.error(
            "Load session error:",
            error
        );

        alert(
            "Unable to load this chat."
        );
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

    if (
        title === null
    ) {
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
                `${API_URL}/chats/${sessionId}`,
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

            throw new Error(
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
                `${API_URL}/chats/${sessionId}`,
                {
                    method: "DELETE"
                }
            );

        if (!response.ok) {

            throw new Error(
                "Unable to delete chat."
            );
        }

        if (
            Number(sessionId) ===
            Number(currentSessionId)
        ) {

            currentSessionId = null;

            localStorage.removeItem(
                "currentSessionId"
            );

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
// MEMORY PANEL
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
                `${API_URL}/memory`
            );

        if (!response.ok) {

            throw new Error(
                "Unable to load memory."
            );
        }

        const data =
            await response.json();

        const memories =
            data.memories || [];

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
                    document.createElement(
                        "div"
                    );

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
            data.documents || [];

        documentList.innerHTML = "";

        if (!documents.length) {

            documentList.innerHTML = `
                <div class="empty-documents">
                    No PDF documents uploaded.
                </div>
            `;

            return;
        }

        documents.forEach(
            function (doc) {

                const item =
                    document.createElement(
                        "div"
                    );

                item.className =
                    "document-item";

                const filename =
                    document.createElement(
                        "span"
                    );

                filename.className =
                    "document-name";

                filename.textContent =
                    doc.filename ||
                    "Unknown PDF";

                const deleteBtn =
                    document.createElement(
                        "button"
                    );

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
                `✓ ${file.name} uploaded successfully`;
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
                `${API_URL}/documents/${encodeURIComponent(
                    filename
                )}`,
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
// SUGGESTION BUTTONS
// ============================================================

function attachSuggestionEvents() {

    document
        .querySelectorAll(
            ".suggestion-btn"
        )
        .forEach(function (button) {

            button.addEventListener(
                "click",
                function () {

                    userInput.value =
                        button.textContent.trim();

                    autoResize();

                    userInput.focus();

                }
            );

        });
}


// ============================================================
// AUTO RESIZE TEXTAREA
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
// SEND BUTTON
// ============================================================

if (sendBtn) {

    sendBtn.addEventListener(
        "click",
        sendMessage
    );
}


// ============================================================
// STOP BUTTON
// ============================================================

if (stopBtn) {

    stopBtn.addEventListener(
        "click",
        stopGenerating
    );
}


// ============================================================
// ENTER KEY
// ============================================================

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


// ============================================================
// NEW CHAT
// ============================================================

if (newChatBtn) {

    newChatBtn.addEventListener(
        "click",
        async function () {

            await createNewChat();

        }
    );
}


// ============================================================
// CLEAR CHAT
// ============================================================

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


// ============================================================
// MEMORY OPEN
// ============================================================

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


// ============================================================
// MEMORY CLOSE
// ============================================================

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


// ============================================================
// PDF UPLOAD BUTTON
// ============================================================

if (uploadBtn) {

    uploadBtn.addEventListener(
        "click",
        uploadPDF
    );
}


// ============================================================
// INITIALIZE APP
// ============================================================

async function initializeApp() {

    try {

        // ----------------------------------------------------
        // Backend health
        // ----------------------------------------------------

        const healthResponse =
            await fetch(
                `${API_URL}/health`
            );

        if (!healthResponse.ok) {

            throw new Error(
                "Backend unavailable"
            );
        }

        statusText.textContent =
            "● Online";

        // ----------------------------------------------------
        // Load chat list
        // ----------------------------------------------------

        await loadChatHistoryList();

        // ----------------------------------------------------
        // Existing session
        // ----------------------------------------------------

        if (currentSessionId) {

            try {

                await loadSession(
                    currentSessionId
                );

            } catch (error) {

                console.warn(
                    "Saved session unavailable.",
                    error
                );

                currentSessionId = null;

                localStorage.removeItem(
                    "currentSessionId"
                );
            }
        }

        // ----------------------------------------------------
        // Create first chat
        // ----------------------------------------------------

        if (!currentSessionId) {

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

    } catch (error) {

        console.error(
            "Initialization error:",
            error
        );

        statusText.textContent =
            "● Backend Offline";
    }
}


// ============================================================
// START APPLICATION
// ============================================================

initializeApp();