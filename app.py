import os
import logging
import requests
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter
from streamlit_pdf_viewer import pdf_viewer
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. SILENCE THE TERMINAL WARNINGS
# ==========================================
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==========================================
# 2. SETUP THE WEBSITE PAGE
# ==========================================
st.set_page_config(
    page_title="Bishop A.A Mayungbo Ministry AI",
    page_icon="✝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# GLOBAL STYLES
# ==========================================
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Playfair+Display:wght@600;700&display=swap');

/* ── CSS Variables ── */
:root {
    --bg-deep:       #0b0d14;
    --bg-panel:      #12151f;
    --bg-card:       #181c2a;
    --bg-input:      #1e2335;
    --gold-primary:  #e8c06a;
    --gold-light:    #f5dfa0;
    --gold-dim:      #a07c3a;
    --purple-glow:   #6c63ff;
    --blue-glow:     #4fa3e0;
    --text-primary:  #f0eee8;
    --text-secondary:#b0a99a;
    --text-muted:    #6e6a62;
    --border:        rgba(232,192,106,0.15);
    --border-hover:  rgba(232,192,106,0.40);
    --radius-sm:     8px;
    --radius-md:     14px;
    --radius-lg:     22px;
}

/* ── Base Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit chrome (keep sidebar toggle) ── */
#MainMenu,
footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton,
[title="View source on GitHub"],
[title="Fork on GitHub"],
a[href*="streamlit.io"],
.viewerBadge_container__1QSob,
.styles_viewerBadge__1yB5_ { display: none !important; }

/* ── Style the sidebar collapse toggle button ── */
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--gold-primary) !important;
}
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="collapsedControl"] button:hover {
    border-color: var(--gold-primary) !important;
    background: rgba(232,192,106,0.08) !important;
}

/* ── Main container ── */
[data-testid="stMain"] {
    background: var(--bg-deep) !important;
    padding-top: 0 !important;
}
.main .block-container {
    padding: 1.5rem 2rem 4rem !important;
    max-width: 860px !important;
    margin: 0 auto !important;
}

/* ── HERO HEADER ── */
.hero-wrap {
    position: relative;
    text-align: center;
    padding: 3rem 1rem 2.5rem;
    overflow: hidden;
    margin-bottom: 0.5rem;
}
.hero-glow {
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 340px;
    background: radial-gradient(ellipse at center,
        rgba(232,192,106,0.18) 0%,
        rgba(108,99,255,0.10) 45%,
        transparent 75%);
    pointer-events: none;
    z-index: 0;
}
.hero-cross {
    font-size: 2.6rem;
    display: block;
    margin-bottom: 0.4rem;
    position: relative;
    z-index: 1;
    filter: drop-shadow(0 0 18px rgba(232,192,106,0.7));
    animation: pulseGlow 3s ease-in-out infinite;
}
@keyframes pulseGlow {
    0%,100% { filter: drop-shadow(0 0 14px rgba(232,192,106,0.6)); }
    50%      { filter: drop-shadow(0 0 28px rgba(232,192,106,0.95)); }
}
.hero-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: clamp(1.6rem, 4vw, 2.6rem) !important;
    font-weight: 700 !important;
    color: var(--gold-light) !important;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin: 0 0 0.5rem !important;
    position: relative; z-index: 1;
    text-shadow: 0 0 40px rgba(232,192,106,0.35);
}
.hero-sub {
    font-size: 1rem !important;
    color: var(--text-secondary) !important;
    font-weight: 400;
    position: relative; z-index: 1;
    letter-spacing: 0.01em;
}
.hero-divider {
    width: 80px; height: 2px;
    margin: 1.2rem auto 0;
    background: linear-gradient(90deg, transparent, var(--gold-primary), transparent);
    border-radius: 2px;
    position: relative; z-index: 1;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.4rem 0 !important;
    border: none !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
}

.stChatMessage [data-testid="stMarkdownContainer"] p {
    font-size: 0.96rem !important;
    line-height: 1.7 !important;
    color: var(--text-primary) !important;
}

/* Avatar tweaks */
[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, var(--purple-glow), var(--blue-glow)) !important;
    border-radius: 50% !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, var(--gold-dim), var(--gold-primary)) !important;
    border-radius: 50% !important;
}

/* Message content cards */
[data-testid="stChatMessage"] > div:last-child {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 2px 20px rgba(0,0,0,0.3) !important;
    backdrop-filter: blur(8px);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:last-child {
    background: linear-gradient(135deg, #1a1e30, #1e2238) !important;
    border-color: rgba(108,99,255,0.25) !important;
}

/* ── Chat input ── */
[data-testid="stChatInputContainer"] {
    background: var(--bg-panel) !important;
    border-top: 1px solid var(--border) !important;
    padding: 1rem 2rem 1.5rem !important;
    position: sticky !important;
    bottom: 0;
    backdrop-filter: blur(12px);
}
[data-testid="stChatInputContainer"] textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    resize: none !important;
    transition: border-color 0.2s ease;
}
[data-testid="stChatInputContainer"] textarea:focus {
    border-color: var(--gold-primary) !important;
    box-shadow: 0 0 0 3px rgba(232,192,106,0.12) !important;
    outline: none !important;
}
[data-testid="stChatInputContainer"] textarea::placeholder {
    color: var(--text-muted) !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, var(--gold-dim), var(--gold-primary)) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: #0b0d14 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
    filter: brightness(1.15) !important;
    transform: scale(1.05) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: var(--gold-primary) !important;
}
[data-testid="stSpinner"] > div {
    border-top-color: var(--gold-primary) !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.25rem !important;
}

/* Sidebar glow header */
.sidebar-header {
    text-align: center;
    padding: 0.5rem 0 1.2rem;
    position: relative;
}
.sidebar-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 50%;
    transform: translateX(-50%);
    width: 60px; height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold-primary), transparent);
}
.sidebar-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.2rem !important;
    color: var(--gold-light) !important;
    font-weight: 600 !important;
    margin: 0 !important;
    text-shadow: 0 0 20px rgba(232,192,106,0.4);
}
.sidebar-subtitle {
    font-size: 0.75rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.2rem !important;
}

/* Sidebar section labels */
.section-label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold-dim) !important;
    padding: 0.8rem 0 0.4rem;
    display: block;
}

/* Sidebar search */
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
    border-color: var(--gold-primary) !important;
    box-shadow: 0 0 0 2px rgba(232,192,106,0.15) !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder {
    color: var(--text-muted) !important;
}

/* Expanders in sidebar */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    margin-bottom: 0.5rem !important;
    transition: border-color 0.2s ease;
    overflow: hidden;
}
[data-testid="stSidebar"] [data-testid="stExpander"]:hover {
    border-color: var(--border-hover) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-size: 0.83rem !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.65rem 0.9rem !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    color: var(--gold-light) !important;
}

/* Download buttons */
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--gold-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
    background: rgba(232,192,106,0.08) !important;
    border-color: var(--gold-primary) !important;
}

/* Admin text inputs */
[data-testid="stSidebar"] [data-testid="stTextArea"] textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}

/* Save button */
[data-testid="stSidebar"] [data-testid="stButton"] button {
    background: linear-gradient(135deg, #2a1f08, #4a3510) !important;
    border: 1px solid var(--gold-dim) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--gold-light) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #3a2a0c, #5a4018) !important;
    border-color: var(--gold-primary) !important;
    box-shadow: 0 0 14px rgba(232,192,106,0.2) !important;
}

/* Alerts & status */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}
.stSuccess {
    background: rgba(52,168,83,0.12) !important;
    border-color: rgba(52,168,83,0.3) !important;
    color: #7ed99a !important;
}
.stError {
    background: rgba(220,53,69,0.12) !important;
    border-color: rgba(220,53,69,0.3) !important;
    color: #f08090 !important;
}
.stWarning {
    background: rgba(232,192,106,0.10) !important;
    border-color: rgba(232,192,106,0.25) !important;
    color: var(--gold-light) !important;
}

/* Dividers */
hr, [data-testid="stDivider"] {
    border-color: var(--border) !important;
    margin: 1rem 0 !important;
}

/* Caption */
.stCaption, [data-testid="stCaptionContainer"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
    text-align: center;
    letter-spacing: 0.03em;
}

/* Subheader overrides inside sidebar */
[data-testid="stSidebar"] h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--gold-dim) !important;
    margin-top: 1rem !important;
    margin-bottom: 0.4rem !important;
}

/* Audio player */
[data-testid="stSidebar"] audio {
    width: 100% !important;
    border-radius: var(--radius-sm) !important;
    margin: 0.5rem 0 !important;
    filter: invert(0.85) hue-rotate(180deg) brightness(0.9);
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--text-muted);
}
.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 0.75rem;
    display: block;
    opacity: 0.5;
    filter: drop-shadow(0 0 12px rgba(232,192,106,0.3));
}
.empty-state-text {
    font-size: 0.9rem;
    color: var(--text-muted);
    line-height: 1.6;
}

/* No results badge */
.no-results {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-align: center;
    padding: 0.75rem;
    background: var(--bg-card);
    border-radius: var(--radius-sm);
    border: 1px dashed var(--border);
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold-primary); }

/* ── Responsive ── */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem 1rem 5rem !important;
    }
    .hero-wrap { padding: 2rem 0.5rem 1.5rem; }
    [data-testid="stChatInputContainer"] { padding: 0.75rem 1rem 1rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HERO HEADER
# ==========================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-glow"></div>
    <span class="hero-cross">✝</span>
    <h1 class="hero-title">Bishop A.A Mayungbo<br>Ministry AI</h1>
    <p class="hero-sub">Ask me anything about faith, the Bible, and spiritual growth.</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. DYNAMIC MEDIA LIBRARY & ADMIN SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <p class="sidebar-title">Ministry Library</p>
        <p class="sidebar-subtitle">Teachings & Resources</p>
    </div>
    """, unsafe_allow_html=True)

    # --- SEARCH BOX ---
    search_query = st.text_input(
        label="search_box",
        placeholder="Search books & sermons...",
        label_visibility="collapsed"
    )

    # --- MEDIA FILES ---
    if os.path.exists("media"):
        media_files = os.listdir("media")
        pdfs = [f for f in media_files if f.lower().endswith(".pdf")]
        audios = [f for f in media_files if f.lower().endswith(".mp3")]

        # Filter by search
        if search_query:
            pdfs = [f for f in pdfs if search_query.lower() in f.lower()]
            audios = [f for f in audios if search_query.lower() in f.lower()]

        if pdfs:
            st.markdown('<span class="section-label">E-Books</span>', unsafe_allow_html=True)
            for pdf in pdfs:
                display_name = pdf.replace("_", " ").replace(".pdf", "")
                with st.expander(f"{display_name}"):
                    pdf_viewer(f"media/{pdf}", width=260)
                    with open(f"media/{pdf}", "rb") as f:
                        st.download_button(
                            label="Download PDF",
                            data=f,
                            file_name=pdf,
                            mime="application/pdf",
                            use_container_width=True
                        )
            if search_query and not pdfs:
                st.markdown('<p class="no-results">No e-books match your search.</p>', unsafe_allow_html=True)

        if audios:
            st.markdown('<span class="section-label">Sermons</span>', unsafe_allow_html=True)
            for audio in audios:
                display_name = audio.replace("_", " ").replace(".mp3", "")
                with st.expander(f"{display_name}"):
                    st.audio(f"media/{audio}")
                    with open(f"media/{audio}", "rb") as f:
                        st.download_button(
                            label="Download Audio",
                            data=f,
                            file_name=audio,
                            mime="audio/mpeg",
                            use_container_width=True
                        )
            if search_query and not audios:
                st.markdown('<p class="no-results">No sermons match your search.</p>', unsafe_allow_html=True)

        if not pdfs and not audios and not search_query:
            st.markdown("""
            <div class="empty-state">
                <span class="empty-state-icon">📭</span>
                <p class="empty-state-text">No media files found.<br>Add PDFs or MP3s to the <strong>media/</strong> folder.</p>
            </div>
            """, unsafe_allow_html=True)

        if search_query and not pdfs and not audios:
            st.markdown(f'<p class="no-results">No results for "{search_query}"</p>', unsafe_allow_html=True)

    st.divider()

    # --- ADMIN PANEL ---
    st.markdown('<span class="section-label">Ministry Admin</span>', unsafe_allow_html=True)
    admin_pass = st.text_input("Admin access", placeholder="Enter admin password", type="password", label_visibility="collapsed")

    if admin_pass == "bishop2024":
        st.success("Admin access granted.")

        st.markdown('<span class="section-label">Teach the AI</span>', unsafe_allow_html=True)
        admin_q = st.text_input("Member's question", placeholder="What did the member ask?", label_visibility="collapsed")
        admin_a = st.text_area("Bishop's answer", placeholder="What is the Bishop's answer?", label_visibility="collapsed")

        if st.button("Save to AI Memory"):
            if admin_q and admin_a:
                with st.spinner("Teaching the AI..."):
                    new_knowledge = f"Question: {admin_q}\nAnswer: {admin_a}"
                    vectorstore.add_texts(
                        texts=[new_knowledge],
                        metadatas=[{"source": "Bishop's Direct Answer"}]
                    )
                st.success("The AI has learned this.")
            else:
                st.warning("Please fill in both the question and the answer.")
    elif admin_pass:
        st.error("Incorrect password.")

    st.divider()
    st.caption("Powered by AI · Trained on Bishop A.A Mayungbo's Teachings")


# ==========================================
# 4. LOAD AI BRAIN & MEMORY
# ==========================================
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
index_name = "preacher-books"

@st.cache_resource
def load_ai_system():
    llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.4)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key)
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    return llm, retriever, vs

llm, retriever, vectorstore = load_ai_system()


# ==========================================
# 5. TELEGRAM ALERT FUNCTION
# ==========================================
def send_telegram_alert(question_text):
    if not telegram_token or not telegram_chat_id:
        return
    message = (
        "🔔 *NEW QUESTION FOR THE BISHOP*\n\n"
        "A member asked a question not in the library:\n\n"
        f"❓ *{question_text}*\n\n"
        "Please reply via the Admin Panel on the website."
    )
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram error: {e}")


# ==========================================
# 6. THE PROMPT
# ==========================================
template = """
You are the official digital assistant and representative of Bishop A.A Mayungbo. 
Your tone is warm, deeply spiritual, encouraging, and filled with the grace of God. You frequently use biblical and charismatic terms of endearment such as "Beloved", "Calvary greetings", "Man/Woman of God", "Hallelujah", and "By his grace".

Your primary source of truth is the provided context. 

RULES:
1. ALWAYS answer using the provided context. 
2. Start your response with a warm spiritual greeting like "Calvary greetings, beloved!" or "Grace and peace to you."
3. Read the Chat History to understand the flow of conversation.
4. If the answer is NOT in the context, you MUST reply EXACTLY with: "Calvary greetings, beloved. That specific question is not currently in my library of Bishop A.A Mayungbo's teachings. I have noted it down and will present it to the ministry team for you!"
5. NEVER break character. 
6. If you find the answer in the context, suggest the user to read the source material in the sidebar.

Chat History:
{chat_history}

Context from the ebooks and sermons:
{context}

User's Question: 
{question}

Helpful Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    formatted = []
    for doc in docs:
        source_file = doc.metadata.get("source", "Unknown Material")
        clean_name = (
            os.path.basename(source_file)
            .replace("_", " ").replace(".pdf", "")
            .replace(".mp3", "").replace(".txt", "")
        )
        formatted.append(f"--- Excerpt from: {clean_name} ---\n{doc.page_content}")
    return "\n\n".join(formatted)

rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history")
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ==========================================
# 7. BUILD THE CHAT INTERFACE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome state
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state" style="padding: 2rem 1rem;">
        <span class="empty-state-icon" style="font-size:2.2rem; opacity:0.7;">✝</span>
        <p class="empty-state-text" style="color: var(--text-secondary);">
            Begin with a question about faith, the Word,<br>or Bishop Mayungbo's teachings.
        </p>
    </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("Ask a question about faith or the teachings…"):
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        with st.spinner("Searching the teachings…"):
            history_str = ""
            for msg in st.session_state.messages[:-1]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"

            response = rag_chain.invoke({
                "question": user_question,
                "chat_history": history_str
            })
            st.markdown(response)

            if "not currently in my library" in response:
                send_telegram_alert(user_question)

    st.session_state.messages.append({"role": "assistant", "content": response})
