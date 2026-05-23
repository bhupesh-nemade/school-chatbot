from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
import time
from uuid import uuid4

import streamlit as st

from chatbot.chain import ask_question


st.set_page_config(page_title="School Chatbot", layout="wide")

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"


LIGHT_MODE_CSS = """
    .stApp {
        background: #f7f8fb;
        color: #172033;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 248, 251, 0.94);
        border-bottom: 1px solid rgba(15, 23, 42, 0.08);
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid rgba(15, 23, 42, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #172033;
    }

    [data-testid="stSidebar"] button {
        background: #f8fafc;
        border-color: rgba(15, 23, 42, 0.1);
        color: #172033;
    }

    [data-testid="stSidebar"] button:hover {
        background: #eef6ff;
        border-color: rgba(14, 165, 233, 0.34);
        color: #0f172a;
    }

    .brand-panel,
    .empty-history,
    .source-item {
        background: #ffffff;
        border-color: rgba(15, 23, 42, 0.1);
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
    }

    .brand-title,
    h1,
    h2,
    h3,
    p,
    label,
    .message-content,
    .source-title {
        color: #172033;
    }

    .brand-subtitle,
    .sidebar-label,
    .empty-history,
    [data-testid="stCaptionContainer"],
    .source-meta,
    details summary {
        color: #64748b;
    }

    .message-bubble.assistant {
        background: #ffffff;
        border-color: rgba(15, 23, 42, 0.08);
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
    }

    .message-bubble.user {
        background: #eaf4ff;
        border-color: rgba(14, 165, 233, 0.18);
    }

   [data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInputContainer"] {
    background: #f7f8fb !important;
}

   [data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    color: #172033 !important;
    border: 1px solid rgba(15, 23, 42, 0.1) !important;
    box-shadow: none !important;
}

    [data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8;
    }
  


    .main div[data-testid="stButton"] button {
        background: #172033;
        border-color: #172033;
        color: #ffffff;
    }

    .main div[data-testid="stButton"] button:hover {
        background: #0f172a;
        border-color: #0f172a;
        color: #ffffff;
    }

    details.sources-panel {
        background: #ffffff;
        border-color: rgba(15, 23, 42, 0.1);
    }
"""

theme_override_css = ""
if st.session_state.theme_mode == "Light":
    theme_override_css = LIGHT_MODE_CSS
elif st.session_state.theme_mode == "System":
    theme_override_css = f"""
    @media (prefers-color-scheme: light) {{
        {LIGHT_MODE_CSS}
    }}
    """


st.markdown(
    """
    <style>
    :root {
        --page-bg: #0f141f;
        --sidebar-bg: #151c2b;
        --surface: #1d2638;
        --surface-2: #243049;
        --surface-user: #2d3f61;
        --surface-assistant: #1f2a40;
        --border: rgba(148, 163, 184, 0.18);
        --border-strong: rgba(148, 163, 184, 0.26);
        --text-main: #f8fafc;
        --text-muted: #a8b3c7;
        --accent: #38bdf8;
    }

    .stApp {
        background: var(--page-bg);
        color: var(--text-main);
    }

    .main .block-container {
        max-width: 1040px;
        padding-top: 2rem;
        padding-bottom: 8.5rem;
    }

    [data-testid="stHeader"] {
        background: rgba(15, 20, 31, 0.94);
        border-bottom: 1px solid rgba(148, 163, 184, 0.10);
    }

    [data-testid="stToolbar"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text-main);
    }

    [data-testid="stSidebar"] button {
        width: 100%;
        min-height: 2.55rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: rgba(36, 48, 73, 0.72);
        color: var(--text-main);
        text-align: left;
        transition: background 150ms ease, border-color 150ms ease, transform 150ms ease;
    }

    [data-testid="stSidebar"] button:hover {
        background: rgba(56, 189, 248, 0.12);
        border-color: rgba(56, 189, 248, 0.42);
        transform: translateY(-1px);
    }

    .brand-panel {
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: var(--surface);
    }

    .brand-title {
        font-size: 1.2rem;
        font-weight: 750;
        margin-bottom: 0.3rem;
        color: var(--text-main);
    }

    .brand-subtitle {
        color: var(--text-muted);
        font-size: 0.88rem;
        line-height: 1.45;
    }

    .sidebar-label {
        color: var(--text-muted);
        font-size: 0.74rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        margin: 1rem 0 0.5rem;
        text-transform: uppercase;
    }

    .empty-history {
        border: 1px dashed var(--border-strong);
        border-radius: 12px;
        color: var(--text-muted);
        font-size: 0.88rem;
        line-height: 1.45;
        padding: 0.8rem;
        background: rgba(36, 48, 73, 0.56);
    }

    h1, h2, h3, p, label {
        color: var(--text-main);
    }

    [data-testid="stCaptionContainer"] {
        color: var(--text-muted);
    }

    .message-row {
        display: flex;
        margin: 0.85rem 0;
    }

    .message-row.user {
        justify-content: flex-end;
    }

    .message-row.assistant {
        justify-content: flex-start;
    }

    .message-bubble {
        max-width: min(760px, 86%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.85rem 1rem;
        line-height: 1.58;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12);
    }

    .message-bubble.user {
        background: var(--surface-user);
        border-color: rgba(125, 211, 252, 0.20);
        border-bottom-right-radius: 6px;
    }

    .message-bubble.assistant {
        background: var(--surface-assistant);
        border-color: rgba(148, 163, 184, 0.22);
        border-bottom-left-radius: 6px;
    }

    .message-content {
        color: var(--text-main);
        font-size: 0.98rem;
        white-space: pre-wrap;
    }

    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stChatInput"] {
        background: var(--page-bg);
    }

    [data-testid="stChatInput"] {
        border-top: 1px solid rgba(148, 163, 184, 0.10);
    }

    [data-testid="stChatInput"] textarea {
        min-height: 56px;
        border-radius: 18px;
        border: 1px solid var(--border-strong);
        background: var(--surface);
        color: var(--text-main);
        padding: 0.9rem 3.6rem 0.9rem 1rem;
        font-size: 0.98rem;
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.24);
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(56, 189, 248, 0.56);
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.24), 0 0 0 1px rgba(56, 189, 248, 0.28);
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-muted);
    }
 
 
 [data-testid="stChatInputSubmitButton"] {
    width: 42px !important;
    height: 42px !important;
    border-radius: 12px !important;
    background: rgba(148, 163, 184, 0.18) !important;
    border: none !important;
    opacity: 1 !important;
}


/* generating state */
body:has(.thinking-inline)
[data-testid="stChatInputSubmitButton"] {
    background: #ef4444 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* HIDE original arrow */
body:has(.thinking-inline)
[data-testid="stChatInputSubmitButton"] svg {
    display: none !important;
}

/* show stop icon */
body:has(.thinking-inline)
[data-testid="stChatInputSubmitButton"]::before {
    content: "■";
    color: white;
    font-size: 22px;
    font-weight: 900;
    line-height: 1;
}   

div[data-testid="column"]:last-child button {
    width: 100%;
    height: 56px;
    border-radius: 16px;
    border: 1px solid rgba(56, 189, 248, 0.4);
    background: var(--accent);
    color: #07111f;
    font-size: 1.2rem;
    font-weight: 800;
}

div[data-testid="column"]:last-child button:hover {
    background: #7dd3fc;
    border-color: #7dd3fc;
}
    details.sources-panel {
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(29, 38, 56, 0.6);
        margin: 0.5rem 0 0.85rem;
        overflow: hidden;
    }

    details.sources-panel summary {
        color: var(--text-muted);
        font-weight: 650;
        padding: 0.65rem 0.8rem;
        cursor: pointer;
    }

    .source-item {
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.75rem;
        margin-bottom: 0.55rem;
        background: rgba(36, 48, 73, 0.78);
    }

    .source-title {
        font-weight: 650;
        color: var(--text-main);
    }

    .source-meta {
        color: var(--text-muted);
        font-size: 0.9rem;
    }
    
    .thinking-inline {
    display: inline-flex;
    align-items: center;
    font-size: 1rem;
    color: var(--text-muted);
    padding: 0.2rem 0;
    background: transparent;
    border: none;
    box-shadow: none;
}

    .thinking-dots {
        display: inline-flex;
        gap: 0.28rem;
        margin-left: 0.35rem;
        vertical-align: middle;
    }

    .thinking-dots span {
        width: 0.36rem;
        height: 0.36rem;
        border-radius: 999px;
        background: currentColor;
        opacity: 0.35;
        animation: thinkingPulse 1.15s infinite ease-in-out;
    }

    .thinking-dots span:nth-child(2) {
        animation-delay: 0.16s;
    }

    .thinking-dots span:nth-child(3) {
        animation-delay: 0.32s;
    }

    @keyframes thinkingPulse {
        0%, 80%, 100% {
            opacity: 0.28;
            transform: translateY(0);
        }
        40% {
            opacity: 1;
            transform: translateY(-2px);
        }
    }
    """
    + theme_override_css
    + """
    </style>
    """,
    unsafe_allow_html=True,
)


def initial_messages():
    return [
        {
            "role": "assistant",
            "content": (
                "Hello. Ask me about admissions, academics, fees, hostel, "
                "transport, library, health, safety, or other school documents."
            ),
            "sources": [],
        }
    ]


def initialize_state():
    if "messages" not in st.session_state:
        st.session_state.messages = initial_messages()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "recent_questions" not in st.session_state:
        st.session_state.recent_questions = []
    if "executor" not in st.session_state:
        st.session_state.executor = ThreadPoolExecutor(max_workers=2)
    if "active_response" not in st.session_state:
        st.session_state.active_response = None
    if "stopped_responses" not in st.session_state:
        st.session_state.stopped_responses = set()


def format_source(doc):
    source = doc.metadata.get("source", "Unknown source")
    page = doc.metadata.get("page", "Unknown")
    source_name = Path(str(source)).name
    return {
        "source": escape(source_name),
        "page": escape(str(page)),
        "preview": escape(doc.page_content[:350].strip()),
    }


def render_sources(sources):
    if not sources:
        return

    with st.expander("Sources used", expanded=False):
        for source in sources:
            st.markdown(f"**{source['source']}**")
            st.caption(f"Page: {source['page']}")
            if source["preview"]:
                st.caption(source["preview"])
            st.divider()


def render_message(message):
    role = message["role"]
    content = escape(message["content"]).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="message-row {role}">
            <div class="message-bubble {role}">
                <div class="message-content">{content}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_sources(message.get("sources", []))

def run_chat_request(question, chat_history):
    try:
        answer, docs = ask_question(question, chat_history)

        fallback_message = "I do not have information related to your question."

        sources = []

        if answer.strip() != fallback_message:
            sources = [format_source(doc) for doc in docs]

        return {
            "answer": answer,
            "sources": sources,
        }

    except Exception as exc:
        return {
            "answer": f"I could not process that question right now. Error: {exc}",
            "sources": [],
        }

def stop_active_response():
    active_response = st.session_state.active_response
    if not active_response:
        return
    request_id = active_response["id"]
    active_response["future"].cancel()
    st.session_state.stopped_responses.add(request_id)
    st.session_state.active_response = None
    st.session_state.executor.shutdown(wait=False, cancel_futures=True)
    st.session_state.executor = ThreadPoolExecutor(max_workers=2)


def submit_question(question):
    stop_active_response()
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    if question not in st.session_state.recent_questions:
        st.session_state.recent_questions.insert(0, question)
    request_id = str(uuid4())
    future = st.session_state.executor.submit(
        run_chat_request,
        question,
        list(st.session_state.chat_history),
    )
    st.session_state.active_response = {
        "id": request_id,
        "question": question,
        "future": future,
    }


def render_active_response():
    active_response = st.session_state.active_response
    if not active_response:
        return False

    request_id = active_response["id"]
    future = active_response["future"]

    stop_clicked = st.session_state.get("active_chat_input")

    if stop_clicked is not None:
        stop_active_response()
        st.rerun()

    if future.done():
        result = future.result()
        st.session_state.active_response = None

        if request_id not in st.session_state.stopped_responses:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }
            )

            if result["sources"]:
                st.session_state.chat_history.append(
                    (active_response["question"], result["answer"])
                )

        st.rerun()

    st.markdown(
        """
        <div class="message-row assistant">
            <div class="thinking-inline">
                Thinking
                <span class="thinking-dots">
                    <span></span><span></span><span></span>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return True


def render_composer():
    if st.session_state.active_response:
        st.chat_input(
            "Generating response...",
            disabled=False,
            key="active_chat_input"
        )

    else:
        question = st.chat_input(
            "Ask a question about the school documents",
            key="normal_chat_input"
        )

        if question and question.strip():
            submit_question(question.strip())
            st.rerun()
            
initialize_state()

with st.sidebar:
    st.markdown(
        """
        <div class="brand-panel">
            <div class="brand-title">School Chatbot</div>
            <div class="brand-subtitle">Document assistant for admissions, academics, fees, transport, hostel, and school policies.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("+ New chat", use_container_width=True):
        stop_active_response()
        st.session_state.messages = initial_messages()
        st.session_state.chat_history = []
        st.rerun()

    st.markdown('<div class="sidebar-label">Appearance</div>', unsafe_allow_html=True)
    st.radio(
        "Mode",
        ["Dark", "Light", "System"],
        key="theme_mode",
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">Chat history</div>', unsafe_allow_html=True)
    if st.session_state.recent_questions:
        for index, question in enumerate(st.session_state.recent_questions[:8]):
            label = question if len(question) <= 42 else f"{question[:39]}..."
            if st.button(label, key=f"history-{index}", use_container_width=True):
                submit_question(question)
                st.rerun()
    else:
        st.markdown(
            """
            <div class="empty-history">
                Your recent questions will appear here after you start asking.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-label">Try asking</div>', unsafe_allow_html=True)
    example_questions = [
        "What is the school location?",
        "What are the admission requirements?",
        "Tell me about transport facilities.",
        "What are the hostel rules?",
    ]
    for example in example_questions:
        if st.button(example, key=f"example-{example}", use_container_width=True):
            submit_question(example)
            st.rerun()

st.title("School Chatbot")
st.caption("Ask questions and get answers from the indexed school documents.")

for message in st.session_state.messages:
    render_message(message)

is_waiting_for_answer = render_active_response()
render_composer()

if is_waiting_for_answer:
    time.sleep(0.8)
    st.rerun()
