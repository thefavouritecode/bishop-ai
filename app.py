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

# ── 1. SILENCE WARNINGS ──────────────────────────────────────────────────────
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── 2. PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bishop A.A Mayungbo Ministry AI",
    page_icon="✝️",
    layout="wide",
    menu_items={}
)

# ── 3. GLOBAL CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=Playfair+Display:wght@600;700&display=swap');

:root {
  --bg:         #090b12;
  --bg-panel:   #0f111a;
  --bg-card:    #141724;
  --bg-input:   #1a1e2e;
  --gold:       #e8c06a;
  --gold-light: #f5dfa0;
  --gold-dim:   #8a6828;
  --purple:     #6c63ff;
  --text:       #ede9e1;
  --text-2:     #9e968a;
  --text-3:     #5a5650;
  --line:       rgba(232,192,106,0.13);
  --line-h:     rgba(232,192,106,0.35);
  --r:          10px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
  height: 100%;
}

/* kill ALL default streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebar"],
.stDeployButton,
[data-testid="stHamburgerButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
a[href*="streamlit.io"],
.viewerBadge_container__1QSob { display: none !important; }

/* full-bleed block container */
.main .block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── APP SHELL ── */
.app-shell {
  display: grid;
  grid-template-columns: 300px 1fr;
  grid-template-rows: 100vh;
  overflow: hidden;
}
@media (max-width: 820px) {
  .app-shell { grid-template-columns: 1fr; grid-template-rows: auto 1fr; }
}

/* ── LEFT PANEL ── */
.panel {
  background: var(--bg-panel);
  border-right: 1px solid var(--line);
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}
.panel-head {
  padding: 1.5rem 1.25rem 1rem;
  border-bottom: 1px solid var(--line);
  flex-shrink: 0;
}
.panel-logo {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.75rem;
}
.panel-cross {
  font-size: 1.4rem;
  filter: drop-shadow(0 0 10px rgba(232,192,106,0.7));
  animation: glow 3s ease-in-out infinite;
}
@keyframes glow {
  0%,100% { filter: drop-shadow(0 0 8px rgba(232,192,106,0.5)); }
  50%      { filter: drop-shadow(0 0 18px rgba(232,192,106,1)); }
}
.panel-name {
  font-family: 'Playfair Display', serif;
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--gold-light);
  line-height: 1.2;
  text-shadow: 0 0 20px rgba(232,192,106,0.3);
}
.panel-name span {
  display: block;
  font-family: 'DM Sans', sans-serif;
  font-size: 0.65rem;
  font-weight: 500;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-3);
  margin-top: 1px;
}
.panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 0.75rem 1.1rem 1rem;
  scrollbar-width: thin;
  scrollbar-color: var(--gold-dim) transparent;
}
.panel-body::-webkit-scrollbar { width: 3px; }
.panel-body::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 3px; }

.sec-label {
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--gold-dim);
  padding: 0.9rem 0 0.45rem;
  display: block;
}
.media-card {
  background: var(--bg-card);
  border: 1px solid var(--line);
  border-radius: var(--r);
  padding: 0.65rem 0.85rem;
  margin-bottom: 0.4rem;
  cursor: pointer;
  transition: border-color 0.18s, background 0.18s;
}
.media-card:hover { border-color: var(--line-h); background: #1b1f30; }
.media-card-name {
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.media-card-type {
  font-size: 0.65rem;
  color: var(--text-3);
  margin-top: 2px;
}
.no-match {
  font-size: 0.75rem;
  color: var(--text-3);
  text-align: center;
  padding: 0.6rem;
  background: var(--bg-card);
  border-radius: var(--r);
  border: 1px dashed var(--line);
}
.panel-foot {
  padding: 0.75rem 1.1rem;
  border-top: 1px solid var(--line);
  font-size: 0.65rem;
  color: var(--text-3);
  text-align: center;
  letter-spacing: 0.04em;
  flex-shrink: 0;
}

/* ── RIGHT CHAT AREA ── */
.chat-area {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--bg);
  overflow: hidden;
}
.chat-topbar {
  padding: 1rem 1.5rem;
  border-bottom: 1px solid var(--line);
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-shrink: 0;
  background: var(--bg-panel);
}
.topbar-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: #3ddc84;
  box-shadow: 0 0 6px #3ddc84;
  flex-shrink: 0;
}
.topbar-label {
  font-size: 0.82rem;
  color: var(--text-2);
}
.topbar-label strong { color: var(--text); font-weight: 600; }

/* ── MESSAGES SCROLL AREA ── */
.chat-messages-wrap {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem 1.5rem 0.5rem;
  scrollbar-width: thin;
  scrollbar-color: var(--gold-dim) transparent;
}
.chat-messages-wrap::-webkit-scrollbar { width: 3px; }
.chat-messages-wrap::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 3px; }

/* compact message bubbles */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0.25rem 0 !important;
  max-width: 720px;
  margin: 0 auto;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
  font-size: 0.875rem !important;
  line-height: 1.65 !important;
  color: var(--text) !important;
  margin: 0 !important;
}
/* Avatar circles */
[data-testid="stChatMessageAvatarUser"] {
  background: linear-gradient(135deg, var(--purple), #4fa3e0) !important;
  border-radius: 50% !important;
  width: 28px !important; height: 28px !important;
  min-width: 28px !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
  background: linear-gradient(135deg, var(--gold-dim), var(--gold)) !important;
  border-radius: 50% !important;
  width: 28px !important; height: 28px !important;
  min-width: 28px !important;
}
/* Message content */
[data-testid="stChatMessage"] > div:last-child {
  background: var(--bg-card) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r) !important;
  padding: 0.6rem 0.9rem !important;
  box-shadow: 0 1px 8px rgba(0,0,0,0.25) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:last-child {
  background: #161929 !important;
  border-color: rgba(108,99,255,0.2) !important;
}

/* ── INPUT AREA ── */
[data-testid="stChatInputContainer"] {
  background: var(--bg-panel) !important;
  border-top: 1px solid var(--line) !important;
  padding: 0.75rem 1.5rem 1rem !important;
  max-width: 100% !important;
  flex-shrink: 0;
}
[data-testid="stChatInputContainer"] textarea {
  background: var(--bg-input) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.875rem !important;
}
[data-testid="stChatInputContainer"] textarea:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px rgba(232,192,106,0.1) !important;
  outline: none !important;
}
[data-testid="stChatInputContainer"] textarea::placeholder { color: var(--text-3) !important; }
[data-testid="stChatInputSubmitButton"] button {
  background: linear-gradient(135deg, var(--gold-dim), var(--gold)) !important;
  border: none !important; border-radius: 8px !important;
  color: #090b12 !important;
  transition: filter 0.15s, transform 0.15s !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
  filter: brightness(1.15) !important; transform: scale(1.06) !important;
}

/* ── SIDEBAR WIDGETS (inside left col) ── */
/* text inputs */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  background: var(--bg-input) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.8rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 2px rgba(232,192,106,0.1) !important;
  outline: none !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder { color: var(--text-3) !important; }
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label { font-size: 0.72rem !important; color: var(--text-3) !important; }

/* expanders */
[data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r) !important;
  margin-bottom: 0.4rem !important;
  overflow: hidden;
  transition: border-color 0.18s;
}
[data-testid="stExpander"]:hover { border-color: var(--line-h) !important; }
[data-testid="stExpander"] summary {
  font-size: 0.78rem !important; font-weight: 500 !important;
  color: var(--text) !important; padding: 0.55rem 0.8rem !important;
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stExpander"] summary:hover { color: var(--gold-light) !important; }

/* buttons */
[data-testid="stDownloadButton"] button,
[data-testid="stButton"] button {
  background: transparent !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r) !important;
  color: var(--gold) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.75rem !important;
  font-weight: 500 !important;
  width: 100% !important;
  transition: all 0.18s !important;
}
[data-testid="stDownloadButton"] button:hover,
[data-testid="stButton"] button:hover {
  background: rgba(232,192,106,0.07) !important;
  border-color: var(--gold) !important;
}

/* alerts */
[data-testid="stAlert"] {
  border-radius: var(--r) !important;
  font-size: 0.78rem !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stSuccess { background: rgba(61,220,132,0.08) !important; border-color: rgba(61,220,132,0.25) !important; color: #7ed9a8 !important; }
.stError   { background: rgba(220,60,60,0.09) !important; border-color: rgba(220,60,60,0.25) !important; color: #f09090 !important; }
.stWarning { background: rgba(232,192,106,0.09) !important; border-color: rgba(232,192,106,0.22) !important; color: var(--gold-light) !important; }

/* spinner */
[data-testid="stSpinner"] > div { border-top-color: var(--gold) !important; }

/* divider */
hr, [data-testid="stDivider"] { border-color: var(--line) !important; margin: 0.75rem 0 !important; }

/* audio */
audio { width: 100% !important; border-radius: var(--r) !important; filter: invert(0.8) hue-rotate(180deg); }

/* ── WELCOME ── */
.welcome {
  text-align: center;
  padding: 3rem 1rem 1.5rem;
  max-width: 420px;
  margin: 0 auto;
}
.welcome-cross {
  font-size: 2.2rem;
  filter: drop-shadow(0 0 14px rgba(232,192,106,0.6));
  animation: glow 3s ease-in-out infinite;
  display: block;
  margin-bottom: 0.75rem;
}
.welcome-title {
  font-family: 'Playfair Display', serif;
  font-size: clamp(1.3rem, 3vw, 1.8rem);
  font-weight: 700;
  color: var(--gold-light);
  text-shadow: 0 0 30px rgba(232,192,106,0.3);
  margin-bottom: 0.5rem;
}
.welcome-sub {
  font-size: 0.85rem;
  color: var(--text-2);
  line-height: 1.6;
}
.welcome-divider {
  width: 50px; height: 1px;
  margin: 1rem auto;
  background: linear-gradient(90deg, transparent, var(--gold), transparent);
}

/* scrollbar global */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 3px; }

/* column gap zero */
[data-testid="stHorizontalBlock"] { gap: 0 !important; }
[data-testid="stColumn"] { padding: 0 !important; }

@media (max-width: 820px) {
  .panel { height: auto; max-height: 40vh; border-right: none; border-bottom: 1px solid var(--line); }
  .chat-area { height: 60vh; }
}
</style>
""", unsafe_allow_html=True)

# ── 4. LOAD AI ────────────────────────────────────────────────────────────────
load_dotenv()
pinecone_api_key  = os.getenv("PINECONE_API_KEY")
groq_api_key      = os.getenv("GROQ_API_KEY")
telegram_token    = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id  = os.getenv("TELEGRAM_CHAT_ID")
index_name        = "preacher-books"

@st.cache_resource
def load_ai_system():
    llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.4)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key)
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    return llm, retriever, vs

llm, retriever, vectorstore = load_ai_system()

# ── 5. TELEGRAM ───────────────────────────────────────────────────────────────
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
    try:
        requests.post(url, data={"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"})
    except Exception as e:
        print(f"Telegram error: {e}")

# ── 6. PROMPT & CHAIN ─────────────────────────────────────────────────────────
template = """
You are the official digital assistant and representative of Bishop A.A Mayungbo.
Your tone is warm, deeply spiritual, encouraging, and filled with the grace of God.
You frequently use biblical and charismatic terms of endearment such as "Beloved",
"Calvary greetings", "Man/Woman of God", "Hallelujah", and "By his grace".

Your primary source of truth is the provided context.

RULES:
1. ALWAYS answer using the provided context.
2. Start your response with a warm spiritual greeting like "Calvary greetings, beloved!" or "Grace and peace to you."
3. Read the Chat History to understand the flow of conversation.
4. If the answer is NOT in the context, you MUST reply EXACTLY with:
   "Calvary greetings, beloved. That specific question is not currently in my library of
   Bishop A.A Mayungbo's teachings. I have noted it down and will present it to the ministry team for you!"
5. NEVER break character.
6. If you find the answer in the context, suggest the user to read the source material in the library panel.

Chat History:
{chat_history}

Context from the ebooks and sermons:
{context}

User's Question:
{question}

Helpful Answer:
"""
prompt_tmpl = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    out = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        name = os.path.basename(src).replace("_", " ").replace(".pdf","").replace(".mp3","").replace(".txt","")
        out.append(f"--- Excerpt from: {name} ---\n{doc.page_content}")
    return "\n\n".join(out)

rag_chain = (
    {
        "context":      itemgetter("question") | retriever | format_docs,
        "question":     itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt_tmpl
    | llm
    | StrOutputParser()
)

# ── 7. SESSION STATE ──────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "open_media"    not in st.session_state: st.session_state.open_media    = None

# ── 8. TWO-COLUMN LAYOUT ──────────────────────────────────────────────────────
left_col, right_col = st.columns([300, 700], gap="small")

# ╔══════════════════════════════════════════╗
# ║  LEFT — LIBRARY PANEL                   ║
# ╚══════════════════════════════════════════╝
with left_col:
    # Header
    st.markdown("""
    <div class="panel-head" style="padding:1.2rem 0 0.8rem;">
      <div class="panel-logo">
        <span class="panel-cross">✝</span>
        <div class="panel-name">
          Bishop A.A Mayungbo
          <span>Ministry Library</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Search
    search_query = st.text_input(
        "search",
        placeholder="🔍  Search books & sermons…",
        label_visibility="collapsed"
    )

    # Media
    if os.path.exists("media"):
        media_files = os.listdir("media")
        pdfs   = sorted([f for f in media_files if f.lower().endswith(".pdf")])
        audios = sorted([f for f in media_files if f.lower().endswith(".mp3")])
        if search_query:
            q = search_query.lower()
            pdfs   = [f for f in pdfs   if q in f.lower()]
            audios = [f for f in audios if q in f.lower()]

        if pdfs:
            st.markdown('<span class="sec-label">E-Books</span>', unsafe_allow_html=True)
            for pdf in pdfs:
                display = pdf.replace("_", " ").replace(".pdf", "")
                with st.expander(f"📄  {display}"):
                    pdf_viewer(f"media/{pdf}", width=240)
                    with open(f"media/{pdf}", "rb") as fh:
                        st.download_button("Download PDF", fh, file_name=pdf,
                                           mime="application/pdf", use_container_width=True)

        if audios:
            st.markdown('<span class="sec-label">Sermons</span>', unsafe_allow_html=True)
            for audio in audios:
                display = audio.replace("_", " ").replace(".mp3", "")
                with st.expander(f"🎙  {display}"):
                    st.audio(f"media/{audio}")
                    with open(f"media/{audio}", "rb") as fh:
                        st.download_button("Download Audio", fh, file_name=audio,
                                           mime="audio/mpeg", use_container_width=True)

        if search_query and not pdfs and not audios:
            st.markdown(f'<p class="no-match">No results for "{search_query}"</p>', unsafe_allow_html=True)

        if not search_query and not pdfs and not audios:
            st.markdown("""
            <p class="no-match" style="margin-top:1rem;">
              Add PDFs or MP3s to the <strong>media/</strong> folder to populate the library.
            </p>""", unsafe_allow_html=True)
    else:
        st.markdown('<p class="no-match" style="margin-top:1rem;">No <code>media/</code> folder found.</p>', unsafe_allow_html=True)

    st.divider()

    # Admin
    st.markdown('<span class="sec-label">🔒  Ministry Admin</span>', unsafe_allow_html=True)
    admin_pass = st.text_input("password", placeholder="Admin password…",
                               type="password", label_visibility="collapsed")
    if admin_pass == "bishop2024":
        st.success("Admin access granted.")
        admin_q = st.text_input("q", placeholder="Member's question…", label_visibility="collapsed")
        admin_a = st.text_area("a", placeholder="Bishop's answer…", label_visibility="collapsed")
        if st.button("💾  Save to AI Memory"):
            if admin_q and admin_a:
                with st.spinner("Teaching the AI…"):
                    vectorstore.add_texts(
                        texts=[f"Question: {admin_q}\nAnswer: {admin_a}"],
                        metadatas=[{"source": "Bishop's Direct Answer"}]
                    )
                st.success("The AI has learned this.")
            else:
                st.warning("Fill in both fields.")
    elif admin_pass:
        st.error("Incorrect password.")

    st.markdown('<p class="panel-foot">Powered by AI · Bishop A.A Mayungbo\'s Teachings</p>',
                unsafe_allow_html=True)


# ╔══════════════════════════════════════════╗
# ║  RIGHT — CHAT PANEL                     ║
# ╚══════════════════════════════════════════╝
with right_col:

    # Top bar
    st.markdown("""
    <div class="chat-topbar">
      <span class="topbar-dot"></span>
      <span class="topbar-label"><strong>Ministry AI</strong> · Ask about faith, the Word, or the Bishop's teachings</span>
    </div>
    """, unsafe_allow_html=True)

    # Welcome screen
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome">
          <span class="welcome-cross">✝</span>
          <div class="welcome-title">Calvary Greetings, Beloved</div>
          <div class="welcome-divider"></div>
          <p class="welcome-sub">
            Ask me anything about faith, the Bible,<br>
            or Bishop A.A Mayungbo's teachings.<br>
            I am here to guide and encourage you.
          </p>
        </div>
        """, unsafe_allow_html=True)

    # Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if user_question := st.chat_input("Ask a question about faith or the teachings…"):
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            with st.spinner("Searching the teachings…"):
                history_str = ""
                for m in st.session_state.messages[:-1]:
                    role = "User" if m["role"] == "user" else "Assistant"
                    history_str += f"{role}: {m['content']}\n"

                response = rag_chain.invoke({
                    "question":     user_question,
                    "chat_history": history_str,
                })
                st.markdown(response)

                if "not currently in my library" in response:
                    send_telegram_alert(user_question)

        st.session_state.messages.append({"role": "assistant", "content": response})
