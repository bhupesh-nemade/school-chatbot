from __future__ import annotations

import html
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import streamlit as st

from chatbot.auth_service import (
    authenticate_google_user,
    authenticate_user,
    register_user,
)
from chatbot.conversation_store import get_conversation_service
from chatbot.rag_service import answer_question
from config import APP_ENV, DEFAULT_MODEL


LOGGER = logging.getLogger(__name__)


# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="School AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# VISUAL THEME
# =============================================================================
# Design language: "Academic Ledger" — a deep-ink / brass-and-parchment palette
# borrowed from campus signage, leather gradebooks and letterhead stationery,
# rather than a generic dashboard look. Fraunces (display serif) carries the
# institutional voice; Inter is the body workhorse; IBM Plex Mono marks
# metadata (pages, timestamps, model tags) the way a card-catalog would.

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root, .stApp {
            color-scheme: light;
        }

        /* Streamlit derives many native-widget colours (tab labels,
           unstyled text, etc.) from its own theme CSS variables rather
           than component classes. Overriding them at the root is what
           actually fixes things like the inactive tab label, instead of
           chasing individual BaseWeb elements one by one. */
        :root {
            --text-color: #16233F;
            --background-color: #FAF6EC;
            --secondary-background-color: #FFFDF7;
            --primary-color: #C79A3D;
        }

        :root {
            --ink: #16233F;
            --ink-soft: #223256;
            --ink-line: #2C3E66;
            --parchment: #FAF6EC;
            --parchment-card: #FFFDF7;
            --rule: #E4DCC5;
            --brass: #C79A3D;
            --brass-dark: #9C7526;
            --sage: #55765B;
            --sage-bg: #E7EFE4;
            --crimson: #9B3E3E;
            --slate: #5B6472;
            --shadow: 0 8px 24px rgba(22, 35, 63, 0.10);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ---------- App background: quiet ruled-paper texture ---------- */
        .stApp {
            background:
                repeating-linear-gradient(
                    to bottom,
                    transparent,
                    transparent 37px,
                    var(--rule) 38px
                ),
                var(--parchment);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--ink) 0%, var(--ink-soft) 100%);
            border-right: 1px solid var(--ink-line);
        }
        [data-testid="stSidebar"] * {
            color: #EFE9D8 !important;
        }
        [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {
            color: #A7B0C7 !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(239, 233, 216, 0.18);
        }

        .crest-header {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.4rem 0 0.9rem 0;
            border-bottom: 1px solid rgba(199, 154, 61, 0.35);
            margin-bottom: 1rem;
        }
        .crest-badge {
            width: 42px;
            height: 42px;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, var(--brass), var(--brass-dark));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            box-shadow: 0 3px 8px rgba(0,0,0,0.35);
            flex-shrink: 0;
        }
        .crest-title {
            font-family: 'Fraunces', serif;
            font-weight: 700;
            font-size: 1.15rem;
            line-height: 1.15;
            letter-spacing: 0.01em;
        }
        .crest-subtitle {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #A7B0C7;
        }

        .user-pill {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(199, 154, 61, 0.14);
            border: 1px solid rgba(199, 154, 61, 0.35);
            border-radius: 999px;
            padding: 0.4rem 0.85rem;
            font-size: 0.82rem;
            margin-bottom: 0.9rem;
        }
        .user-pill .dot {
            width: 7px; height: 7px; border-radius: 50%;
            background: #7FBF87;
            box-shadow: 0 0 6px #7FBF87;
            flex-shrink: 0;
        }

        .sidebar-section-label {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #8C96AE;
            margin: 0.6rem 0 0.35rem 0;
        }

        /* Sidebar buttons */
        [data-testid="stSidebar"] .stButton > button {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(239, 233, 216, 0.16);
            border-radius: 10px;
            text-align: left;
            transition: all 0.15s ease;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(199, 154, 61, 0.16);
            border-color: var(--brass);
            transform: translateX(2px);
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--brass) 0%, var(--brass-dark) 100%);
            border: none;
            color: var(--ink) !important;
            font-weight: 600;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] * {
            color: var(--ink) !important;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            filter: brightness(1.08);
            transform: none;
        }

        /* ---------- Hero header on main page ---------- */
        .hero-card {
            background: linear-gradient(120deg, var(--ink) 0%, var(--ink-soft) 100%);
            border-radius: 16px;
            padding: 1.6rem 2rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.1rem;
            position: relative;
            overflow: hidden;
        }
        .hero-card::after {
            content: "";
            position: absolute;
            top: -40%; right: -8%;
            width: 220px; height: 220px;
            background: radial-gradient(circle, rgba(199,154,61,0.28) 0%, transparent 70%);
        }
        .hero-title {
            font-family: 'Fraunces', serif;
            font-weight: 700;
            font-size: 2rem;
            color: #FBF7EC;
            margin: 0 0 0.25rem 0;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        .hero-subtitle {
            color: #B9C1D6;
            font-size: 0.95rem;
            max-width: 640px;
            line-height: 1.5;
            margin-bottom: 0.9rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: rgba(127, 191, 135, 0.16);
            border: 1px solid rgba(127, 191, 135, 0.45);
            color: #BFE8C4;
            border-radius: 999px;
            padding: 0.28rem 0.8rem;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .status-pill .dot {
            width: 7px; height: 7px; border-radius: 50%;
            background: #7FBF87;
            box-shadow: 0 0 6px #7FBF87;
        }

        /* ---------- Chat bubbles ---------- */
        .msg-row {
            display: flex;
            align-items: flex-end;
            gap: 0.6rem;
            margin: 0.9rem 0;
            animation: fadeInUp 0.28s ease;
        }
        .msg-row.user { justify-content: flex-end; }
        .msg-row.assistant { justify-content: flex-start; }

        .avatar {
            width: 34px; height: 34px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 1rem;
            flex-shrink: 0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.18);
        }
        .avatar.assistant-avatar {
            background: radial-gradient(circle at 30% 30%, var(--brass), var(--brass-dark));
            color: var(--ink);
        }
        .avatar.user-avatar {
            background: var(--ink);
            color: #EFE9D8;
        }

        .bubble {
            max-width: 72%;
            padding: 0.75rem 1.05rem;
            border-radius: 16px;
            line-height: 1.55;
            font-size: 0.96rem;
            box-shadow: 0 2px 10px rgba(22,35,63,0.06);
        }
        .bubble.assistant-bubble {
            background: var(--parchment-card);
            border: 1px solid var(--rule);
            border-bottom-left-radius: 4px;
            color: var(--ink);
        }
        .bubble.user-bubble {
            background: linear-gradient(135deg, var(--ink) 0%, var(--ink-soft) 100%);
            color: #FBF7EC;
            border-bottom-right-radius: 4px;
        }

        .welcome-note {
            font-family: 'Fraunces', serif;
            font-size: 1.02rem;
            font-style: italic;
            color: var(--slate);
        }

        /* ---------- Typing indicator ---------- */
        .typing-row { display: flex; align-items: center; gap: 0.6rem; margin: 0.9rem 0; }
        .typing-bubble {
            background: var(--parchment-card);
            border: 1px solid var(--rule);
            border-radius: 16px;
            border-bottom-left-radius: 4px;
            padding: 0.85rem 1.1rem;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .typing-bubble .dot {
            width: 6px; height: 6px; border-radius: 50%;
            background: var(--brass-dark);
            animation: bounce 1.1s infinite ease-in-out;
        }
        .typing-bubble .dot:nth-child(2) { animation-delay: 0.15s; }
        .typing-bubble .dot:nth-child(3) { animation-delay: 0.3s; }
        .typing-label {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            color: var(--slate);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.55; }
            30% { transform: translateY(-5px); opacity: 1; }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media (prefers-reduced-motion: reduce) {
            .msg-row, .typing-bubble .dot { animation: none !important; }
        }

        /* ---------- Sources / citations ---------- */
        [data-testid="stExpander"] {
            border: 1px solid var(--rule) !important;
            border-radius: 12px !important;
            background: var(--parchment-card) !important;
            box-shadow: 0 1px 4px rgba(22,35,63,0.05);
        }
        .citation-card {
            border-left: 3px solid var(--brass);
            background: rgba(199, 154, 61, 0.06);
            border-radius: 0 8px 8px 0;
            padding: 0.55rem 0.85rem;
            margin-bottom: 0.5rem;
        }
        .citation-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 20px; height: 20px;
            border-radius: 50%;
            background: var(--brass);
            color: var(--ink);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }
        .citation-meta {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: var(--slate);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* ---------- Chat input ----------
           Keep the entire input area visually unified like a modern
           assistant composer: one white surface, one border, one shadow.
           The outer Streamlit footer stays transparent so no dark band
           appears behind the composer. */
        div:has(> [data-testid="stChatInput"]),
        div:has(> div > [data-testid="stChatInput"]),
        div:has(> div > div > [data-testid="stChatInput"]),
        [data-testid="stBottom"],
        [data-testid="stBottomBlockContainer"],
        [data-testid="stChatInputContainer"] {
            background: transparent !important;
            box-shadow: none !important;
        }

        [data-testid="stChatInput"],
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] form {
            background: #FFFFFF !important;
            border: 1px solid #D8DDE7 !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 24px rgba(22, 35, 63, 0.12) !important;
        }

        [data-testid="stChatInput"] {
            padding: 0.15rem !important;
            outline: none !important;
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: var(--brass) !important;
            box-shadow: 0 0 0 3px rgba(199, 154, 61, 0.18),
                        0 8px 24px rgba(22, 35, 63, 0.12) !important;
            outline: none !important;
        }

        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] textarea:focus,
        [data-testid="stChatInput"] [contenteditable="true"] {
            background: #FFFFFF !important;
            color: var(--ink) !important;
            font-family: 'Inter', sans-serif;
            caret-color: var(--ink) !important;
            border: none !important;
            outline: none !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #6B7280 !important;
            opacity: 1 !important;
        }

        [data-testid="stChatInput"] button {
            background: var(--ink) !important;
            border: none !important;
            border-radius: 10px !important;
            box-shadow: none !important;
        }

        [data-testid="stChatInput"] button:hover {
            background: var(--ink-soft) !important;
        }

        [data-testid="stChatInput"] button svg {
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
        }

        [data-testid="stChatInput"] button:disabled {
            background: #E5E7EB !important;
        }

        [data-testid="stChatInput"] button:disabled svg {
            fill: #6B7280 !important;
            stroke: #6B7280 !important;
        }

        /* ---------- Buttons (global) ----------
           Streamlit gives regular buttons kind="primary"/"secondary" but
           form-submit buttons kind="primaryFormSubmit"/"secondaryFormSubmit" —
           both variants need covering or they fall back to Streamlit's
           default red theme. */
        .stButton > button,
        [data-testid="stFormSubmitButton"] button {
            border-radius: 10px;
            font-weight: 600;
            transition: filter 0.15s ease;
        }

        .stButton > button[kind="primary"],
        .stButton > button[kind="primaryFormSubmit"],
        [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
            background: linear-gradient(135deg, var(--brass) 0%, var(--brass-dark) 100%) !important;
            border: none !important;
            color: var(--ink) !important;
        }
        .stButton > button[kind="primary"] *,
        .stButton > button[kind="primaryFormSubmit"] *,
        [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] * {
            color: var(--ink) !important;
        }
        .stButton > button[kind="primary"]:hover,
        .stButton > button[kind="primaryFormSubmit"]:hover {
            filter: brightness(1.06);
        }

        .stButton > button[kind="secondary"],
        .stButton > button[kind="secondaryFormSubmit"],
        [data-testid="stFormSubmitButton"] button[kind="secondaryFormSubmit"] {
            background: var(--parchment-card) !important;
            border: 1px solid var(--rule) !important;
            color: var(--ink) !important;
        }
        .stButton > button[kind="secondary"] *,
        .stButton > button[kind="secondaryFormSubmit"] * {
            color: var(--ink) !important;
        }
        .stButton > button[kind="secondary"]:hover,
        .stButton > button[kind="secondaryFormSubmit"]:hover {
            border-color: var(--brass) !important;
        }

        /* Auth view toggle: a real segmented control built from two
           st.button calls, styled as pills. This replaces st.tabs, whose
           label colours are baked into Streamlit's own generated styles
           and can't be reliably overridden with CSS. */
        .st-key-auth_tab_login button,
        .st-key-auth_tab_register button {
            border-radius: 999px !important;
        }

        /* Text inputs: force a light, legible field regardless of the
           browser/OS colour scheme, with a warm focus glow */
        [data-testid="stTextInput"] input {
            border-radius: 8px !important;
            background: var(--parchment-card) !important;
            color: var(--ink) !important;
            border: 1px solid var(--rule) !important;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: var(--brass) !important;
            box-shadow: 0 0 0 3px rgba(199, 154, 61, 0.22) !important;
            outline: none !important;
        }
        [data-testid="stTextInput"] input::placeholder {
            color: var(--slate) !important;
            opacity: 1 !important;
        }
        [data-testid="stTextInputRootElement"] {
            background: var(--parchment-card) !important;
            border-color: var(--rule) !important;
        }
        [data-testid="stTextInput"] button {
            background: transparent !important;
            color: var(--slate) !important;
        }
        [data-testid="stTextInput"] button:hover {
            color: var(--brass-dark) !important;
        }
        [data-testid="stTextInput"] button svg {
            fill: var(--slate) !important;
            stroke: var(--slate) !important;
            opacity: 1 !important;
        }
        [data-testid="stTextInput"] button:hover svg {
            fill: var(--brass-dark) !important;
            stroke: var(--brass-dark) !important;
        }

        /* Auth card: soft radial glow behind it + a ledger-tab accent
           stripe across the top, so the card reads as one composed
           piece rather than a bare Streamlit form */
        .auth-wrap {
            position: relative;
            padding-top: 0.5rem;
        }
        .auth-wrap::before {
            content: "";
            position: absolute;
            top: -60px; left: 50%;
            transform: translateX(-50%);
            width: 420px; height: 420px;
            background: radial-gradient(circle, rgba(199,154,61,0.16) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }
        [data-testid="stForm"] {
            position: relative;
            overflow: hidden;
            background: var(--parchment-card);
            border: 1px solid var(--rule);
            border-radius: 16px;
            padding: 1.7rem 1.7rem 1.15rem 1.7rem;
            box-shadow: 0 16px 40px rgba(22, 35, 63, 0.14);
        }
        [data-testid="stForm"]::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--brass) 0%, var(--sage) 50%, var(--crimson) 100%);
        }
        [data-testid="stForm"] label,
        [data-testid="stForm"] [data-testid="stWidgetLabel"] p {
            color: var(--ink) !important;
            font-weight: 600 !important;
        }
        .auth-title {
            font-family: 'Fraunces', serif;
            font-size: 2.6rem;
            font-weight: 700;
            text-align: center;
            color: var(--ink);
            margin-top: 2.2rem;
            margin-bottom: 0.15rem;
        }
        .auth-subtitle {
            text-align: center;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--slate);
            margin-bottom: 2rem;
        }
        .auth-divider {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            margin: 1.6rem 0 1rem 0;
        }
        .auth-divider .line {
            flex: 1;
            height: 1px;
            background: var(--rule);
        }
        .auth-divider .label {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--slate);
            white-space: nowrap;
        }

        /* Google sign-in button: real multicolour "G" mark instead of
           an emoji, layered in via a background-image on ::before since
           st.button labels can't render raw HTML/SVG. */
        .st-key-google_login_btn button {
            position: relative;
            padding-left: 2.5rem !important;
        }
        .st-key-google_login_btn button::before {
            content: "";
            position: absolute;
            left: 1.1rem;
            top: 50%;
            transform: translateY(-50%);
            width: 18px;
            height: 18px;
            background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxOCAxOCIgd2lkdGg9IjE4IiBoZWlnaHQ9IjE4Ij4KPHBhdGggZmlsbD0iIzQyODVGNCIgZD0iTTE3LjY0IDkuMjA0NWMwLS42MzgxLS4wNTczLTEuMjUxOC0uMTYzNi0xLjg0MDlIOXYzLjQ4MTRoNC44NDM2Yy0uMjA4NiAxLjEyNS0uODQyNyAyLjA3ODItMS43OTU5IDIuNzE2NHYyLjI1ODFoMi45MDg3YzEuNzAxOC0xLjU2NjggMi42ODM2LTMuODc0MSAyLjY4MzYtNi42MTV6Ii8+CjxwYXRoIGZpbGw9IiMzNEE4NTMiIGQ9Ik05IDE4YzIuNDMgMCA0LjQ2NzMtLjgwNjQgNS45NTY0LTIuMTgxOGwtMi45MDg3LTIuMjU4MWMtLjgwNjQuNTQtMS44MzY4Ljg2MTgtMy4wNDc3Ljg2MTgtMi4zNDU1IDAtNC4zMjgyLTEuNTgzNi01LjAzNTktMy43MTA0SC45NTczdjIuMzMxOEMyLjQzODIgMTUuOTgzMiA1LjQ4MTggMTggOSAxOHoiLz4KPHBhdGggZmlsbD0iI0ZCQkMwNSIgZD0iTTMuOTY0MSAxMC43MWMtLjE4LS41NC0uMjgyMy0xLjExNTUtLjI4MjMtMS43MXMuMTAyMy0xLjE3LjI4MjMtMS43MVY0Ljk1ODJILjk1NzNDLjM0NzcgNi4xNzMyIDAgNy41NDc3IDAgOXMuMzQ3NyAyLjgyNjguOTU3MyA0LjA0MThMMy45NjQxIDEwLjcxeiIvPgo8cGF0aCBmaWxsPSIjRUE0MzM1IiBkPSJNOSAzLjU3OTVjMS4zMjE0IDAgMi41MDc3LjQ1NDEgMy40NDA1IDEuMzQ1OWwyLjU4MTgtMi41ODE4QzEzLjQ2MzIuODkxOCAxMS40MjU5IDAgOSAwIDUuNDgxOCAwIDIuNDM4MiAyLjAxNjguOTU3MyA0Ljk1ODJMMy45NjQxIDcuMjlDNC42NzE4IDUuMTYzMiA2LjY1NDUgMy41Nzk1IDkgMy41Nzk1eiIvPgo8L3N2Zz4K");
            background-size: contain;
            background-repeat: no-repeat;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()


# =============================================================================
# SESSION STATE
# =============================================================================

def initialize_session_state() -> None:
    defaults = {
        "authenticated": False,
        "authenticated_username": "",
        "user_id": None,
        "conversation_id": None,
        "messages": [],
        "chat_history": [],
        "conversations": [],
        "selected_model": DEFAULT_MODEL,
        "active_response": None,
        "stopped_responses": set(),
        "executor": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            if isinstance(value, list):
                st.session_state[key] = []
            elif isinstance(value, set):
                st.session_state[key] = set()
            else:
                st.session_state[key] = value

    if st.session_state.executor is None:
        st.session_state.executor = ThreadPoolExecutor(
            max_workers=2
        )


initialize_session_state()


# =============================================================================
# BASIC HELPERS
# =============================================================================

def get_service():
    return get_conversation_service()


def current_user_id() -> str:
    user_id = st.session_state.get("user_id")

    if not user_id:
        raise RuntimeError(
            "Authenticated user ID is missing."
        )

    return str(user_id)


def current_username() -> str:
    return str(
        st.session_state.get(
            "authenticated_username",
            "",
        )
    )


def sanitize_display_text(
    value: object,
) -> str:
    """
    Remove HTML tags from stored/generated text.

    This prevents old messages containing HTML such as:
    <div>...</div>
    from appearing in the UI.
    """

    text = str(value or "")

    text = re.sub(
        r"<[^>]*>",
        "",
        text,
    )

    return text.strip()


def escape_for_markdown(
    text: str,
) -> str:
    """
    Escape HTML before sending content to Streamlit Markdown.
    """

    return html.escape(
        text,
        quote=False,
    )


# =============================================================================
# AUTHENTICATED SESSION
# =============================================================================

def set_authenticated_session(user) -> None:
    """
    Convert an application User object into Streamlit session state.
    """

    st.session_state.authenticated = True

    st.session_state.authenticated_username = (
        user.email or user.name
    )

    st.session_state.user_id = user.user_id

    st.session_state.selected_model = DEFAULT_MODEL

    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.conversations = []
    st.session_state.active_response = None
    st.session_state.stopped_responses = set()

    old_executor = st.session_state.get("executor")

    if old_executor is not None:
        try:
            old_executor.shutdown(
                wait=False,
                cancel_futures=True,
            )
        except Exception:
            LOGGER.exception(
                "Failed to reset executor."
            )

    st.session_state.executor = ThreadPoolExecutor(
        max_workers=2
    )


# =============================================================================
# GOOGLE OIDC
# =============================================================================

def google_is_logged_in() -> bool:
    """
    Safely detect an authenticated Streamlit OIDC session.

    We intentionally avoid direct access to:
        st.user.is_logged_in

    because that attribute is unavailable when OIDC is not configured
    or when Streamlit has not initialized that field.
    """

    try:
        return bool(
            getattr(
                st.user,
                "is_logged_in",
                False,
            )
        )
    except Exception:
        return False


def handle_google_login() -> None:
    """
    If Streamlit has an authenticated Google/OIDC session,
    map that identity into our SQLite users/user_identities tables.
    """

    if not google_is_logged_in():
        return

    # Do not recreate the local session every rerun.
    if st.session_state.get("authenticated"):
        return

    try:
        identity = st.user.to_dict()

        provider_user_id = str(
            identity.get("sub", "") or ""
        ).strip()

        email = str(
            identity.get("email", "") or ""
        ).strip()

        name = str(
            identity.get("name", "") or ""
        ).strip()

        email_verified = bool(
            identity.get(
                "email_verified",
                False,
            )
        )

        if not provider_user_id:
            raise ValueError(
                "Google did not provide a user identifier."
            )

        if not email:
            raise ValueError(
                "Google did not provide an email address."
            )

        if not name:
            name = email.split("@", 1)[0]

        user = authenticate_google_user(
            provider_user_id=provider_user_id,
            email=email,
            name=name,
            email_verified=email_verified,
        )

        set_authenticated_session(user)

    except ValueError as exc:
        LOGGER.exception(
            "Google identity validation failed."
        )

        st.error(str(exc))

        try:
            st.logout()
        except Exception:
            LOGGER.exception(
                "Failed to logout invalid Google session."
            )

        st.stop()

    except Exception:
        LOGGER.exception(
            "Google authentication failed."
        )

        if APP_ENV == "development":
            st.error(
                "Google login failed. "
                "Check the Streamlit terminal for the traceback."
            )
        else:
            st.error(
                "Unable to sign in with Google right now."
            )

        st.stop()


# Handle Google session before showing the local login page.
handle_google_login()


# =============================================================================
# LOGOUT
# =============================================================================

def logout_user() -> None:
    active = st.session_state.get(
        "active_response"
    )

    if active:
        future = active.get("future")

        if future:
            try:
                future.cancel()
            except Exception:
                LOGGER.exception(
                    "Failed to cancel active response during logout."
                )

    executor = st.session_state.get(
        "executor"
    )

    if executor:
        try:
            executor.shutdown(
                wait=False,
                cancel_futures=True,
            )
        except Exception:
            LOGGER.exception(
                "Failed to shutdown executor during logout."
            )

    st.session_state.authenticated = False
    st.session_state.authenticated_username = ""
    st.session_state.user_id = None
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.conversations = []
    st.session_state.active_response = None
    st.session_state.stopped_responses = set()

    st.session_state.executor = ThreadPoolExecutor(
        max_workers=2
    )

    # If this is an OIDC/Google session, clear the Streamlit identity too.
    if google_is_logged_in():
        try:
            st.logout()
            return
        except Exception:
            LOGGER.exception(
                "Failed to logout from Streamlit OIDC."
            )

    st.rerun()


# =============================================================================
# LOGIN / REGISTRATION UI
# =============================================================================

def render_auth_page() -> None:
    st.markdown(
        '<div class="auth-title">🎓 School AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="auth-subtitle">'
        "School knowledge assistant"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)

    if "auth_view" not in st.session_state:
        st.session_state.auth_view = "login"

    _, center, _ = st.columns([1, 1.4, 1])

    with center:

        toggle_login_col, toggle_register_col = st.columns(2)

        with toggle_login_col:
            if st.button(
                "Login",
                use_container_width=True,
                type=(
                    "primary"
                    if st.session_state.auth_view == "login"
                    else "secondary"
                ),
                key="auth_tab_login",
            ):
                st.session_state.auth_view = "login"
                st.rerun()

        with toggle_register_col:
            if st.button(
                "Create account",
                use_container_width=True,
                type=(
                    "primary"
                    if st.session_state.auth_view == "register"
                    else "secondary"
                ),
                key="auth_tab_register",
            ):
                st.session_state.auth_view = "register"
                st.rerun()

        # ---------------------------------------------------------------
        # EMAIL / PASSWORD LOGIN
        # ---------------------------------------------------------------

        if st.session_state.auth_view == "login":

            with st.form(
                "login_form",
                clear_on_submit=False,
            ):
                email = st.text_input(
                    "Email",
                    placeholder="you@example.com",
                    autocomplete="email",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    autocomplete="current-password",
                )

                submitted = st.form_submit_button(
                    "Login",
                    use_container_width=True,
                    type="primary",
                )

            if submitted:
                try:
                    user = authenticate_user(
                        email=email,
                        password=password,
                    )

                    if user is None:
                        st.error(
                            "Invalid email or password."
                        )
                        return

                    set_authenticated_session(user)
                    st.rerun()

                except ValueError as exc:
                    st.error(str(exc))

                except Exception:
                    LOGGER.exception(
                        "Login failed."
                    )

                    if APP_ENV == "development":
                        st.error(
                            "Login failed. "
                            "Check the Streamlit terminal."
                        )
                    else:
                        st.error(
                            "Unable to log in right now."
                        )

        # ---------------------------------------------------------------
        # REGISTRATION
        # ---------------------------------------------------------------

        else:

            with st.form(
                "register_form",
                clear_on_submit=False,
            ):
                name = st.text_input(
                    "Full name",
                    placeholder="Your full name",
                )

                register_email = st.text_input(
                    "Email",
                    placeholder="you@example.com",
                    autocomplete="email",
                )

                register_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="At least 8 characters",
                    autocomplete="new-password",
                )

                confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                    placeholder="Repeat your password",
                    autocomplete="new-password",
                )

                submitted = st.form_submit_button(
                    "Create account",
                    use_container_width=True,
                    type="primary",
                )

            if submitted:

                if register_password != confirm_password:
                    st.error(
                        "Passwords do not match."
                    )
                    return

                try:
                    user = register_user(
                        name=name,
                        email=register_email,
                        password=register_password,
                    )

                    set_authenticated_session(user)

                    st.success(
                        "Account created successfully."
                    )

                    st.rerun()

                except ValueError as exc:
                    st.error(str(exc))

                except Exception:
                    LOGGER.exception(
                        "Registration failed."
                    )

                    if APP_ENV == "development":
                        st.error(
                            "Registration failed. "
                            "Check the Streamlit terminal."
                        )
                    else:
                        st.error(
                            "Unable to create the account right now."
                        )

        # ---------------------------------------------------------------
        # GOOGLE LOGIN
        # ---------------------------------------------------------------

        st.markdown(
            """
            <div class="auth-divider">
                <span class="line"></span>
                <span class="label">Or continue with</span>
                <span class="line"></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Continue with Google",
            use_container_width=True,
            type="secondary",
            key="google_login_btn",
        ):
            st.login()

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# REQUIRE AUTHENTICATION
# =============================================================================

if not st.session_state.authenticated:
    render_auth_page()
    st.stop()


# =============================================================================
# CONVERSATION STATE
# =============================================================================

def serialize_message(message) -> dict:
    return {
        "role": message.role,
        "content": message.content,
        "sources": message.sources,
    }


def refresh_conversation_state() -> None:
    user_id = current_user_id()

    conversation_id = (
        st.session_state.conversation_id
    )

    if not conversation_id:
        return

    service = get_service()

    st.session_state.messages = [
        serialize_message(message)
        for message in service.get_messages(
            user_id,
            conversation_id,
        )
    ]

    st.session_state.chat_history = (
        service.get_chat_history(
            user_id,
            conversation_id,
        )
    )

    st.session_state.conversations = (
        service.list_conversations(user_id)
    )


def initialize_conversation() -> None:
    user_id = current_user_id()

    conversation = (
        get_service()
        .ensure_latest_conversation(user_id)
    )

    st.session_state.conversation_id = (
        conversation.conversation_id
    )

    refresh_conversation_state()


if not st.session_state.conversation_id:
    initialize_conversation()

elif not st.session_state.messages:
    refresh_conversation_state()


# =============================================================================
# CHAT EXECUTION
# =============================================================================

def run_chat_request(
    question: str,
    chat_history: list,
    model_name: str,
    user_id: str,
    conversation_id: str,
) -> dict:
    """
    Runs in a background worker.

    Never access st.session_state here.
    """

    try:
        return answer_question(
            question=question,
            model_name=model_name,
            chat_history=chat_history,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    except Exception as exc:
        LOGGER.exception(
            "Chat request failed "
            "user_id=%s conversation_id=%s",
            user_id,
            conversation_id,
        )

        if APP_ENV == "development":
            return {
                "answer": (
                    "I couldn't process that request.\n\n"
                    f"Development error: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "sources": [],
            }

        return {
            "answer": (
                "I couldn't process that request "
                "right now. Please try again."
            ),
            "sources": [],
        }


def stop_active_response() -> None:
    active = st.session_state.get(
        "active_response"
    )

    if not active:
        return

    request_id = active.get("id")
    future = active.get("future")

    if future:
        try:
            future.cancel()
        except Exception:
            LOGGER.exception(
                "Failed to cancel active response."
            )

    if request_id:
        st.session_state.stopped_responses.add(
            request_id
        )

    st.session_state.active_response = None


def submit_question(
    question: str,
) -> None:
    question = (
        question or ""
    ).strip()

    if not question:
        return

    stop_active_response()

    service = get_service()

    user_id = current_user_id()

    conversation_id = (
        st.session_state.conversation_id
    )

    model_name = (
        st.session_state.selected_model
    )

    chat_history = (
        service.get_chat_history(
            user_id,
            conversation_id,
        )
    )

    service.add_user_message(
        user_id,
        conversation_id,
        question,
    )

    refresh_conversation_state()

    request_id = str(uuid4())

    future = st.session_state.executor.submit(
        run_chat_request,
        question,
        list(chat_history),
        model_name,
        user_id,
        conversation_id,
    )

    st.session_state.active_response = {
        "id": request_id,
        "conversation_id": conversation_id,
        "future": future,
    }


def render_active_response() -> bool:
    active = st.session_state.get(
        "active_response"
    )

    if not active:
        return False

    future = active["future"]
    request_id = active["id"]

    if future.done():

        try:
            result = future.result()

        except Exception as exc:
            LOGGER.exception(
                "Background response failed."
            )

            if APP_ENV == "development":
                result = {
                    "answer": (
                        "I couldn't process that request.\n\n"
                        f"Development error: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "sources": [],
                }
            else:
                result = {
                    "answer": (
                        "I couldn't process that "
                        "request right now."
                    ),
                    "sources": [],
                }

        st.session_state.active_response = None

        if request_id not in (
            st.session_state.stopped_responses
        ):

            service = get_service()

            user_id = current_user_id()

            conversation_id = (
                active["conversation_id"]
            )

            service.add_assistant_message(
                user_id,
                conversation_id,
                result.get(
                    "answer",
                    "",
                ),
                result.get(
                    "sources",
                    [],
                ),
            )

            refresh_conversation_state()

        st.rerun()

    st.markdown(
        """
        <div class="typing-row">
            <div class="avatar assistant-avatar">🎓</div>
            <div class="typing-bubble">
                <span class="dot"></span><span class="dot"></span><span class="dot"></span>
            </div>
            <span class="typing-label">School&nbsp;AI&nbsp;is&nbsp;consulting&nbsp;the&nbsp;archives&hellip;</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return True


# =============================================================================
# SOURCE RENDERING
# =============================================================================

def render_sources(
    sources: list[dict],
) -> None:
    if not sources:
        return

    with st.expander(
        f"📚 Sources used · {len(sources)}",
        expanded=False,
    ):
        for index, source in enumerate(
            sources,
            start=1,
        ):
            source_name = sanitize_display_text(
                source.get(
                    "source",
                    "Unknown",
                )
            )

            page = sanitize_display_text(
                source.get(
                    "page",
                    "Unknown",
                )
            )

            preview = sanitize_display_text(
                source.get(
                    "preview",
                    "",
                )
            )

            preview_html = (
                f'<div style="margin-top:0.35rem; color: var(--ink);">'
                f'{escape_for_markdown(preview)}</div>'
                if preview
                else ""
            )

            st.markdown(
                f"""
                <div class="citation-card">
                    <span class="citation-badge">{index}</span>
                    <strong>{escape_for_markdown(source_name)}</strong>
                    <div class="citation-meta">Page {escape_for_markdown(page)}</div>
                    {preview_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# MESSAGE RENDERING
# =============================================================================

def render_message(
    message: dict,
) -> None:
    role = message.get(
        "role",
        "assistant",
    )

    raw_content = sanitize_display_text(
        message.get(
            "content",
            "",
        )
    )

    safe_content = escape_for_markdown(
        raw_content
    ).replace("\n", "<br>")

    is_user = role == "user"

    row_class = "user" if is_user else "assistant"
    avatar_class = "user-avatar" if is_user else "assistant-avatar"
    bubble_class = "user-bubble" if is_user else "assistant-bubble"
    avatar_glyph = "🧑‍🎓" if is_user else "🎓"

    if is_user:
        st.markdown(
            f"""
            <div class="msg-row {row_class}">
                <div class="bubble {bubble_class}">{safe_content}</div>
                <div class="avatar {avatar_class}">{avatar_glyph}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="msg-row {row_class}">
                <div class="avatar {avatar_class}">{avatar_glyph}</div>
                <div class="bubble {bubble_class}">{safe_content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_sources(
            message.get(
                "sources",
                [],
            )
        )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown(
        """
        <div class="crest-header">
            <div class="crest-badge">🎓</div>
            <div>
                <div class="crest-title">School AI</div>
                <div class="crest-subtitle">Knowledge Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="user-pill">
            <span class="dot"></span> Signed in as {escape_for_markdown(current_username())}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "🚪 Log out",
        use_container_width=True,
    ):
        logout_user()

    st.divider()

    if st.button(
        "＋ New conversation",
        use_container_width=True,
        type="primary",
    ):
        stop_active_response()

        conversation = (
            get_service()
            .create_conversation(
                current_user_id()
            )
        )

        st.session_state.conversation_id = (
            conversation.conversation_id
        )

        refresh_conversation_state()

        st.rerun()

    st.markdown(
        '<div class="sidebar-section-label">Conversations</div>',
        unsafe_allow_html=True,
    )

    conversations = (
        st.session_state.conversations
    )

    if not conversations:
        st.caption(
            "No conversations yet."
        )

    for conversation in conversations:

        title = (
            conversation.title
            or "New chat"
        )

        if len(title) > 36:
            title = (
                title[:36]
                + "..."
            )

        is_active = (
            conversation.conversation_id
            == st.session_state.conversation_id
        )

        label = (
            f"● {title}"
            if is_active
            else f"  {title}"
        )

        button_container = (
            st.container(border=True)
            if is_active
            else st.container()
        )

        with button_container:
            if st.button(
                label,
                key=(
                    "conversation_"
                    + conversation.conversation_id
                ),
                use_container_width=True,
            ):
                stop_active_response()

                st.session_state.conversation_id = (
                    conversation.conversation_id
                )

                refresh_conversation_state()

                st.rerun()

    st.divider()

    st.markdown(
        '<div class="sidebar-section-label">Model</div>',
        unsafe_allow_html=True,
    )

    st.selectbox(
        "Model",
        [DEFAULT_MODEL],
        key="selected_model",
        label_visibility="collapsed",
    )


# =============================================================================
# MAIN UI
# =============================================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🎓 School AI Assistant</div>
        <div class="hero-subtitle">
            Ask questions about admissions, academics, fees, transport,
            policies, and other indexed school documents — every answer
            is grounded in the source material and cited below it.
        </div>
        <div class="status-pill"><span class="dot"></span> System ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)


display_messages = (
    st.session_state.messages
    or [
        {
            "role": "assistant",
            "content": (
                "Welcome. Ask me anything about "
                "the indexed school documents."
            ),
            "sources": [],
        }
    ]
)

for message in display_messages:
    render_message(message)


# =============================================================================
# ACTIVE REQUEST
# =============================================================================

waiting = render_active_response()

if waiting:
    time.sleep(0.8)
    st.rerun()


# =============================================================================
# CHAT INPUT
# =============================================================================

question = st.chat_input(
    "Ask about your school documents..."
)

if question:
    submit_question(question)
    st.rerun()