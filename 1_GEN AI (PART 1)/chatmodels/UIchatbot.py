#run using:-python -m streamlit run chatmodels/UIchatbot.py
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# ───────────────────────── Existing backend (unchanged) ─────────────────────────
load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"
TEMPERATURE = 0.7
MAX_TOKENS = 300

model = ChatGroq(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

# Same four modes as the terminal version (1 = Angry, 2 = Funny, 3 = Sad, else = Professional)
MODES = {
    "Angry": "You are an angry AI agent. Respond aggressively and impatiently.",
    "Funny": "You are a very funny AI agent. Respond with humor and jokes.",
    "Sad": "You are a sad AI agent. Respond in a sad and emotional way.",
    "Professional": "You are a helpful and professional AI agent.",
}
MODE_ICONS = {"Angry": "😠", "Funny": "😂", "Sad": "😢", "Professional": "💼"}

# ─────────────────────────────── UI layer only ───────────────────────────────
st.set_page_config(
    page_title="AI Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --navy-900: #0A1030;
    --navy-800: #111A45;
    --navy-700: #1B2660;
    --indigo: #3B4BD8;
    --violet: #7C5CFF;
    --ink: #1A2040;
    --muted: #6B7299;
    --mist: #F3F4FC;
    --line: #E3E6F5;
    --card: #FFFFFF;
    --grad: linear-gradient(135deg, #3B4BD8 0%, #7C5CFF 100%);
}

html, body, [class*="css"], .stApp, button, input, textarea {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
}

/* Remove default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

/* Canvas */
.stApp {
    background:
        radial-gradient(900px 420px at 85% -10%, rgba(124,92,255,.14), transparent 60%),
        radial-gradient(800px 380px at 0% 0%, rgba(59,75,216,.10), transparent 55%),
        var(--mist);
    color: var(--ink);
}
.block-container {
    max-width: 860px;
    padding: 1.6rem 1.25rem 7rem 1.25rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy-900) 0%, var(--navy-800) 100%);
    border-right: 1px solid rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
.sb-brand { display:flex; align-items:center; gap:.7rem; margin-bottom:1.6rem; }
.sb-logo {
    width:40px; height:40px; border-radius:12px; background:var(--grad);
    display:flex; align-items:center; justify-content:center;
    color:#fff; font-size:1.2rem; box-shadow:0 8px 20px rgba(124,92,255,.45);
}
.sb-name { color:#fff; font-weight:700; font-size:1.05rem; line-height:1.1; }
.sb-tag  { color:#9AA3D6; font-size:.74rem; margin-top:2px; }
.sb-section { color:#8F99D4; font-size:.78rem; font-weight:600; margin:1.3rem 0 .55rem .2rem; letter-spacing:.02em; }
.sb-card {
    background: rgba(255,255,255,.05);
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 14px; padding: .8rem .95rem; margin-bottom:.6rem;
}
.sb-label { color:#8F99D4; font-size:.72rem; margin-bottom:.15rem; }
.sb-value { color:#F2F4FF; font-size:.9rem; font-weight:600; word-break:break-word; }
.sb-row { display:flex; gap:.6rem; }
.sb-row .sb-card { flex:1; }

/* ── Mode selector (sidebar radio as cards) ── */
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] { gap: .5rem; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    width: 100%; margin: 0; cursor: pointer;
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.09);
    border-radius: 14px; padding: .65rem .85rem;
    transition: background .15s ease, border-color .15s ease;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.1); }
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(59,75,216,.6), rgba(124,92,255,.6));
    border-color: rgba(176,160,255,.75);
    box-shadow: 0 8px 20px rgba(91,80,235,.3);
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #E8EBFF !important; font-weight: 600; font-size: .92rem;
}

/* ── Header ── */
.chat-header {
    display:flex; align-items:center; justify-content:space-between;
    background: var(--card); border:1px solid var(--line); border-radius:20px;
    padding: 1rem 1.25rem; margin-bottom: 1.4rem;
    box-shadow: 0 1px 2px rgba(20,30,90,.04), 0 12px 32px rgba(40,50,140,.08);
}
.ch-left { display:flex; align-items:center; gap:.9rem; }
.ch-logo {
    width:46px; height:46px; border-radius:14px; background:var(--grad);
    display:flex; align-items:center; justify-content:center;
    color:#fff; font-size:1.35rem; box-shadow:0 8px 18px rgba(91,80,235,.35);
}
.ch-title { font-weight:800; font-size:1.2rem; color:var(--navy-800); letter-spacing:-.01em; line-height:1.15; }
.ch-sub   { font-size:.82rem; color:var(--muted); margin-top:3px; }
.ch-chips { display:flex; align-items:center; gap:.5rem; }
.ch-mode {
    font-size:.8rem; font-weight:600; color:var(--indigo); background:#ECEEFF;
    border:1px solid #D8DCFB; padding:.35rem .75rem; border-radius:999px;
}
.ch-status {
    display:flex; align-items:center; gap:.45rem; font-size:.8rem; font-weight:600;
    color:#1F8F5F; background:#E8F7F0; border:1px solid #CBEEDD;
    padding:.35rem .75rem; border-radius:999px;
}
.ch-dot { width:8px; height:8px; border-radius:50%; background:#26B97A; box-shadow:0 0 0 3px rgba(38,185,122,.2); }

/* ── Empty state ── */
.empty {
    text-align:center; padding: 3.5rem 1rem 2rem;
}
.empty-orb {
    width:72px; height:72px; margin:0 auto 1.1rem; border-radius:22px; background:var(--grad);
    display:flex; align-items:center; justify-content:center; color:#fff; font-size:2rem;
    box-shadow:0 18px 40px rgba(91,80,235,.35);
}
.empty h2 { font-size:1.5rem; font-weight:800; color:var(--navy-800); margin:0 0 .35rem; letter-spacing:-.01em; }
.empty p  { color:var(--muted); font-size:.95rem; margin:0; }

/* ── Messages ── */
[data-testid="stChatMessage"] {
    background: transparent; border: none; padding: .35rem 0; gap: .75rem;
    align-items: flex-start;
}
[data-testid="stChatMessageContent"] {
    border-radius: 18px; padding: .85rem 1.1rem; max-width: 82%;
    font-size: .96rem; line-height: 1.6;
}
[data-testid="stChatMessageContent"] p { margin: 0 0 .5rem; }
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }

/* hidden role markers */
[data-testid="stMarkdownContainer"]:has(> .a-msg), [data-testid="stMarkdownContainer"]:has(> .u-msg) {
    display: none;
}

/* AI bubble */
[data-testid="stChatMessage"]:has(.a-msg) [data-testid="stChatMessageContent"] {
    background: var(--card); border: 1px solid var(--line); color: var(--ink);
    border-top-left-radius: 6px;
    box-shadow: 0 1px 2px rgba(20,30,90,.04), 0 8px 22px rgba(40,50,140,.07);
}
/* User bubble — right aligned */
[data-testid="stChatMessage"]:has(.u-msg) {
    flex-direction: row-reverse;
}
[data-testid="stChatMessage"]:has(.u-msg) [data-testid="stChatMessageContent"] {
    background: var(--grad); color: #fff; border-top-right-radius: 6px;
    box-shadow: 0 10px 26px rgba(91,80,235,.32);
}
[data-testid="stChatMessage"]:has(.u-msg) [data-testid="stChatMessageContent"] * {
    color: #fff !important;
}

/* Avatars */
[data-testid="stChatMessageAvatarAssistant"], [data-testid="stChatMessageAvatarUser"] {
    width: 36px; height: 36px; border-radius: 12px; font-size: 1rem;
    display:flex; align-items:center; justify-content:center;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: var(--grad) !important; color:#fff !important;
    box-shadow: 0 6px 16px rgba(91,80,235,.3);
}
[data-testid="stChatMessageAvatarUser"] {
    background: #E7E9FA !important; border: 1px solid var(--line);
}

/* ── Chat input ── */
[data-testid="stBottom"], [data-testid="stBottom"] > div { background: transparent !important; }
[data-testid="stBottom"] > div { padding-bottom: 1.1rem; }
[data-testid="stChatInput"] {
    max-width: 860px; margin: 0 auto;
    background: var(--card); border: 1px solid var(--line); border-radius: 999px;
    padding: .25rem .4rem .25rem .8rem;
    box-shadow: 0 2px 4px rgba(20,30,90,.05), 0 18px 44px rgba(40,50,140,.14);
    transition: border-color .15s ease, box-shadow .15s ease;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--violet);
    box-shadow: 0 0 0 4px rgba(124,92,255,.15), 0 18px 44px rgba(40,50,140,.16);
}
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] div[data-baseweb] {
    background: transparent !important; border: none !important; box-shadow: none !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--ink) !important; font-size: .97rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #9AA0C4 !important; }
[data-testid="stChatInput"] button {
    background: var(--grad) !important; border: none !important;
    border-radius: 999px !important; color: #fff !important;
    box-shadow: 0 6px 16px rgba(91,80,235,.4);
}
[data-testid="stChatInput"] button:hover { filter: brightness(1.08); }
[data-testid="stChatInput"] button svg { fill: #fff !important; color: #fff !important; }

/* Spinner */
.stSpinner > div { color: var(--muted) !important; }
.stSpinner svg { stroke: var(--violet) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-thumb { background: #CDD1EC; border-radius: 10px; border: 2px solid var(--mist); }

/* ── Responsive ── */
@media (max-width: 768px) {
    .block-container { padding: 1rem .8rem 7rem; }
    .chat-header { padding: .8rem .9rem; border-radius: 16px; }
    .ch-status { display: none; }
    .ch-logo { width: 40px; height: 40px; }
    [data-testid="stChatMessageContent"] { max-width: 92%; padding: .75rem .95rem; }
    .empty { padding-top: 2rem; }
}

@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ──
with st.sidebar:
    st.markdown(
        """
<div class="sb-brand">
    <div class="sb-logo">✦</div>
    <div>
        <div class="sb-name">AI Assistant</div>
        <div class="sb-tag">Powered by Groq</div>
    </div>
</div>
<div class="sb-section">Choose your AI mode</div>
""",
        unsafe_allow_html=True,
    )
    selected_mode = st.radio(
        "AI mode",
        list(MODES.keys()),
        index=3,
        format_func=lambda m: f"{MODE_ICONS[m]}  {m}",
        label_visibility="collapsed",
    )

# Picking a mode starts a fresh conversation with that system message
# (the terminal version picked the mode once at startup).
if st.session_state.get("active_mode") != selected_mode:
    st.session_state.active_mode = selected_mode
    st.session_state.messages = [SystemMessage(content=MODES[selected_mode])]
messages = st.session_state.messages

with st.sidebar:
    st.markdown(
        f"""
<div class="sb-section">Persona</div>
<div class="sb-card">
    <div class="sb-label">System role</div>
    <div class="sb-value">{MODES[selected_mode]}</div>
</div>

<div class="sb-section">Model</div>
<div class="sb-card">
    <div class="sb-label">Name</div>
    <div class="sb-value">{MODEL_NAME}</div>
</div>
<div class="sb-row">
    <div class="sb-card">
        <div class="sb-label">Temperature</div>
        <div class="sb-value">{TEMPERATURE}</div>
    </div>
    <div class="sb-card">
        <div class="sb-label">Max tokens</div>
        <div class="sb-value">{MAX_TOKENS}</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ── Header ──
st.markdown(
    f"""
<div class="chat-header">
    <div class="ch-left">
        <div class="ch-logo">✦</div>
        <div>
            <div class="ch-title">AI Assistant</div>
            <div class="ch-sub">Currently in {selected_mode.lower()} mode</div>
        </div>
    </div>
    <div class="ch-chips">
        <div class="ch-mode">{MODE_ICONS[selected_mode]} {selected_mode}</div>
        <div class="ch-status"><span class="ch-dot"></span>Online</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

AI_AVATAR = "✨"
USER_AVATAR = "🙂"

# ── Chat history (system message is not shown) ──
has_history = any(not isinstance(m, SystemMessage) for m in messages)

if not has_history:
    st.markdown(
        """
<div class="empty">
    <div class="empty-orb">✦</div>
    <h2>How can I help today?</h2>
    <p>Type a message below to start the conversation.</p>
</div>
""",
        unsafe_allow_html=True,
    )

def show_user(text):
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown('<span class="u-msg"></span>', unsafe_allow_html=True)
        st.markdown(text)


for m in messages:
    if isinstance(m, HumanMessage):
        show_user(m.content)
    elif isinstance(m, AIMessage):
        with st.chat_message("assistant", avatar=AI_AVATAR):
            st.markdown('<span class="a-msg"></span>', unsafe_allow_html=True)
            st.markdown(m.content)

# ── Input + existing model call (logic unchanged) ──
prompt = st.chat_input("Message AI Assistant…")

if prompt:
    messages.append(HumanMessage(content=prompt))
    show_user(prompt)

    with st.chat_message("assistant", avatar=AI_AVATAR):
        st.markdown('<span class="a-msg"></span>', unsafe_allow_html=True)
        with st.spinner("Thinking…"):
            response = model.invoke(messages)
        st.markdown(response.content)

    messages.append(AIMessage(content=response.content))