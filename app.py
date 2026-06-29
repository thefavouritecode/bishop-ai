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

# ── 3. SESSION STATE ──────────────────────────────────────────────────────────
if "messages"       not in st.session_state: st.session_state.messages       = []
if "panel_open"     not in st.session_state: st.session_state.panel_open     = True
if "admin_unlocked" not in st.session_state: st.session_state.admin_unlocked = False

# ── 4. GLOBAL CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=Playfair+Display:wght@600;700&display=swap');

:root {
  --bg:         #0a0b10;
  --bg-panel:   #0e1018;
  --bg-card:    #13161f;
  --bg-input:   #161924;
  --gold:       #e8c06a;
  --gold-light: #f5dfa0;
  --gold-dim:   #7a5e28;
  --gold-glow:  rgba(232,192,106,0.18);
  --purple:     #6c63ff;
  --text:       #edeae2;
  --text-2:     #9a9284;
  --text-3:     #52504a;
  --line:       rgba(232,192,106,0.11);
  --line-h:     rgba(232,192,106,0.32);
  --r:          12px;
  --r-pill:     999px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
}

/* kill all streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stHamburgerButton"],
.stDeployButton,
a[href*="streamlit.io"],
.viewerBadge_container__1QSob { display: none !important; }

/* full-bleed */
.main .block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* zero gap columns */
[data-testid="stHorizontalBlock"] { gap: 0 !important; align-items: stretch !important; }
[data-testid="stColumn"] { padding: 0 !important; }

/* ─────────────────────────────────────────
   LEFT PANEL
───────────────────────────────────────── */
.panel-wrap {
  background: var(--bg-panel);
  border-right: 1px solid var(--line);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.panel-head {
  padding: 1.25rem 1.1rem 0.9rem;
  border-bottom: 1px solid var(--line);
  flex-shrink: 0;
}
.panel-logo-row {
  display: flex;
  align-items: center;
  gap: 0.55rem;
}
.p-cross {
  font-size: 1.35rem;
  filter: drop-shadow(0 0 8px rgba(232,192,106,0.7));
  animation: pglow 3s ease-in-out infinite;
  flex-shrink: 0;
}
@keyframes pglow {
  0%,100% { filter: drop-shadow(0 0 7px rgba(232,192,106,0.55)); }
  50%      { filter: drop-shadow(0 0 16px rgba(232,192,106,0.95)); }
}
.p-name {
  font-family: 'Playfair Display', serif;
  font-size: 0.88rem;
  font-weight: 700;
  color: var(--gold-light);
  line-height: 1.2;
  text-shadow: 0 0 16px rgba(232,192,106,0.28);
}
.p-name span {
  display: block;
  font-family: 'DM Sans', sans-serif;
  font-size: 0.6rem;
  font-weight: 500;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-3);
  margin-top: 2px;
}
.panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 0.6rem 1rem 0.8rem;
  scrollbar-width: thin;
  scrollbar-color: var(--gold-dim) transparent;
}
.panel-body::-webkit-scrollbar { width: 3px; }
.panel-body::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 3px; }
.panel-foot {
  padding: 0.65rem 1rem;
  border-top: 1px solid var(--line);
  font-size: 0.62rem;
  color: var(--text-3);
  text-align: center;
  letter-spacing: 0.04em;
  flex-shrink: 0;
}
.sec-lbl {
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--gold-dim);
  padding: 0.85rem 0 0.38rem;
  display: block;
}
.no-match {
  font-size: 0.72rem;
  color: var(--text-3);
  text-align: center;
  padding: 0.55rem;
  background: var(--bg-card);
  border-radius: var(--r);
  border: 1px dashed var(--line);
  margin-top: 0.5rem;
}

/* expanders inside panel */
[data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r) !important;
  margin-bottom: 0.35rem !important;
  overflow: hidden;
  transition: border-color 0.18s;
}
[data-testid="stExpander"]:hover { border-color: var(--line-h) !important; }
[data-testid="stExpander"] summary {
  font-size: 0.76rem !important; font-weight: 500 !important;
  color: var(--text) !important; padding: 0.5rem 0.75rem !important;
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stExpander"] summary:hover { color: var(--gold-light) !important; }

/* text inputs panel */
[data-testid="stTextInput"] input {
  background: var(--bg-input) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.78rem !important;
  padding: 0.45rem 0.75rem !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 2px rgba(232,192,106,0.1) !important;
  outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--text-3) !important; }
[data-testid="stTextInput"] label { font-size: 0 !important; }

[data-testid="stTextArea"] textarea {
  background: var(--bg-input) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.78rem !important;
}
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 2px rgba(232,192,106,0.1) !important;
  outline: none !important;
}
[data-testid="stTextArea"] label { font-size: 0 !important; }

/* buttons */
[data-testid="stDownloadButton"] button,
[data-testid="stButton"] button {
  background: transparent !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r) !important;
  color: var(--gold) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  width: 100% !important;
  padding: 0.38rem 0.8rem !important;
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
  font-size: 0.74rem !important;
  font-family: 'DM Sans', sans-serif !important;
  padding: 0.45rem 0.75rem !important;
}
.stSuccess { background: rgba(61,220,132,0.08) !important; border-color: rgba(61,220,132,0.22) !important; color: #7ed9a8 !important; }
.stError   { background: rgba(220,60,60,0.09)  !important; border-color: rgba(220,60,60,0.22)  !important; color: #f09090 !important; }
.stWarning { background: rgba(232,192,106,0.09)!important; border-color: rgba(232,192,106,0.2) !important; color: var(--gold-light) !important; }

hr, [data-testid="stDivider"] { border-color: var(--line) !important; margin: 0.65rem 0 !important; }
audio { width: 100% !important; border-radius: var(--r) !important; filter: invert(0.8) hue-rotate(180deg); }

/* ─────────────────────────────────────────
   TOGGLE BUTTON column
───────────────────────────────────────── */
.toggle-wrap {
  background: var(--bg);
  border-right: 1px solid var(--line);
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding-top: 0.85rem;
  height: 100vh;
}
.toggle-btn {
  background: var(--bg-card);
  border: 1px solid var(--line);
  border-radius: 8px;
  width: 28px; height: 28px;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer;
  color: var(--gold);
  font-size: 0.85rem;
  transition: border-color 0.18s, background 0.18s;
  user-select: none;
}
.toggle-btn:hover {
  border-color: var(--gold);
  background: rgba(232,192,106,0.08);
}

/* ─────────────────────────────────────────
   RIGHT CHAT AREA
───────────────────────────────────────── */
.chat-wrap {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--bg);
  overflow: hidden;
}

/* top bar — title only */
.chat-topbar {
  padding: 0.85rem 1.5rem;
  border-bottom: 1px solid var(--line);
  display: flex;
  align-items: center;
  gap: 0.55rem;
  flex-shrink: 0;
  background: var(--bg-panel);
}
.topbar-cross {
  font-size: 1rem;
  filter: drop-shadow(0 0 6px rgba(232,192,106,0.65));
  animation: pglow 3s ease-in-out infinite;
}
.topbar-title {
  font-family: 'Playfair Display', serif;
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--gold-light);
  text-shadow: 0 0 14px rgba(232,192,106,0.25);
}

/* messages scroll */
.msgs-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 1.25rem 0 0.5rem;
  scrollbar-width: thin;
  scrollbar-color: var(--gold-dim) transparent;
}
.msgs-scroll::-webkit-scrollbar { width: 3px; }
.msgs-scroll::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 3px; }

/* compact chat bubbles */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0.2rem 0 !important;
  max-width: 680px;
  margin: 0 auto;
  width: 100%;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
  font-size: 0.875rem !important;
  line-height: 1.65 !important;
  color: var(--text) !important;
  margin: 0 !important;
}
[data-testid="stChatMessageAvatarUser"] {
  background: linear-gradient(135deg, var(--purple), #4fa3e0) !important;
  border-radius: 50% !important; width: 26px !important; height: 26px !important; min-width: 26px !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
  background: linear-gradient(135deg, var(--gold-dim), var(--gold)) !important;
  border-radius: 50% !important; width: 26px !important; height: 26px !important; min-width: 26px !important;
}
[data-testid="stChatMessage"] > div:last-child {
  background: var(--bg-card) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r) !important;
  padding: 0.6rem 0.9rem !important;
  box-shadow: 0 1px 6px rgba(0,0,0,0.22) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:last-child {
  background: #13152a !important;
  border-color: rgba(108,99,255,0.18) !important;
}

/* ─────────────────────────────────────────
   INPUT — pill with glow
───────────────────────────────────────── */
.input-zone {
  flex-shrink: 0;
  padding: 0.85rem 1.5rem 1.1rem;
  background: var(--bg);
  display: flex;
  justify-content: center;
}
[data-testid="stChatInputContainer"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  max-width: 640px !important;
  width: 100% !important;
  margin: 0 auto !important;
}
[data-testid="stChatInputContainer"] > div {
  background: var(--bg-input) !important;
  border: 1.5px solid rgba(232,192,106,0.35) !important;
  border-radius: var(--r-pill) !important;
  box-shadow:
    0 0 0 4px rgba(232,192,106,0.06),
    0 0 22px rgba(232,192,106,0.12),
    0 0 48px rgba(232,192,106,0.06) !important;
  transition: box-shadow 0.25s, border-color 0.25s !important;
  overflow: hidden;
}
[data-testid="stChatInputContainer"] > div:focus-within {
  border-color: var(--gold) !important;
  box-shadow:
    0 0 0 4px rgba(232,192,106,0.1),
    0 0 28px rgba(232,192,106,0.22),
    0 0 60px rgba(232,192,106,0.10) !important;
}
[data-testid="stChatInputContainer"] textarea {
  background: transparent !important;
  border: none !important;
  border-radius: var(--r-pill) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.875rem !important;
  padding: 0.65rem 1rem !important;
  resize: none !important;
  outline: none !important;
  box-shadow: none !important;
}
[data-testid="stChatInputContainer"] textarea::placeholder { color: var(--text-3) !important; }
[data-testid="stChatInputSubmitButton"] button {
  background: linear-gradient(135deg, #5a4010, var(--gold)) !important;
  border: none !important;
  border-radius: 50% !important;
  width: 32px !important; height: 32px !important;
  margin: 4px 6px 4px 0 !important;
  color: #090a0f !important;
  transition: filter 0.15s, transform 0.15s !important;
  display: flex; align-items: center; justify-content: center;
}
[data-testid="stChatInputSubmitButton"] button:hover {
  filter: brightness(1.2) !important;
  transform: scale(1.08) !important;
}

/* spinner */
[data-testid="stSpinner"] > div { border-top-color: var(--gold) !important; }

/* ─────────────────────────────────────────
   WELCOME SCREEN
───────────────────────────────────────── */
.welcome {
  text-align: center;
  padding: 3.5rem 1rem 1rem;
  max-width: 400px;
  margin: 0 auto;
}
.w-cross {
  font-size: 2.5rem;
  display: block;
  filter: drop-shadow(0 0 18px rgba(232,192,106,0.65));
  animation: pglow 3s ease-in-out infinite;
  margin-bottom: 0.9rem;
}
.w-title {
  font-family: 'Playfair Display', serif;
  font-size: clamp(1.25rem, 2.5vw, 1.65rem);
  font-weight: 700;
  color: var(--gold-light);
  text-shadow: 0 0 28px rgba(232,192,106,0.28);
  line-height: 1.25;
  margin-bottom: 0.55rem;
}
.w-rule {
  width: 44px; height: 1px;
  margin: 0.75rem auto;
  background: linear-gradient(90deg, transparent, var(--gold), transparent);
}
.w-sub {
  font-size: 0.83rem;
  color: var(--text-2);
  line-height: 1.65;
}

/* scrollbar global */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── 5. LOAD AI ────────────────────────────────────────────────────────────────
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key     = os.getenv("GROQ_API_KEY")
telegram_token   = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
index_name       = "preacher-books"

@st.cache_resource
def load_ai_system():
    llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.4)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key)
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    return llm, retriever, vs

llm, retriever, vectorstore = load_ai_system()

# ── 6. TELEGRAM ───────────────────────────────────────────────────────────────
def send_telegram_alert(question_text):
    if not telegram_token or not telegram_chat_id:
        return
    msg = (
        "🔔 *NEW QUESTION FOR THE BISHOP*\n\n"
        "A member asked a question not in the library:\n\n"
        f"❓ *{question_text}*\n\n"
        "Please reply via the Admin Panel on the website."
    )
    try:
        requests.post(
            f"https://api.telegram.org/bot{telegram_token}/sendMessage",
            data={"chat_id": telegram_chat_id, "text": msg, "parse_mode": "Markdown"}
        )
    except Exception as e:
        print(f"Telegram error: {e}")

# ── 7. PROMPT & CHAIN ─────────────────────────────────────────────────────────
template = """
You are the official digital assistant and representative of Bishop A.A Mayungbo.
Your tone is warm, deeply spiritual, encouraging, and filled with the grace of God.
You frequently use terms like "Beloved", "Calvary greetings", "Man/Woman of God", "Hallelujah", "By his grace".

RULES:
1. ALWAYS answer using the provided context.
2. Start with a warm spiritual greeting like "Calvary greetings, beloved!" or "Grace and peace to you."
3. Read the Chat History to understand the conversation flow.
4. If the answer is NOT in the context, reply EXACTLY:
   "Calvary greetings, beloved. That specific question is not currently in my library of Bishop A.A Mayungbo's teachings. I have noted it down and will present it to the ministry team for you!"
5. NEVER break character.
6. If you find the answer, suggest the user read the source material in the library panel.

Chat History:
{chat_history}

Context:
{context}

User's Question:
{question}

Helpful Answer:
"""
prompt_tmpl = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    out = []
    for doc in docs:
        src  = doc.metadata.get("source", "Unknown")
        name = os.path.basename(src).replace("_"," ").replace(".pdf","").replace(".mp3","").replace(".txt","")
        out.append(f"--- Excerpt from: {name} ---\n{doc.page_content}")
    return "\n\n".join(out)

rag_chain = (
    {
        "context":      itemgetter("question") | retriever | format_docs,
        "question":     itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt_tmpl | llm | StrOutputParser()
)

# ── 8. TOGGLE BUTTON ──────────────────────────────────────────────────────────
# Thin toggle column + optional panel column + chat column
ICON_OPEN  = "‹"   # arrow pointing left  (panel visible → click to hide)
ICON_CLOSE = "›"   # arrow pointing right (panel hidden  → click to show)

toggle_col, panel_col, chat_col = st.columns(
    [28, 290, 682] if st.session_state.panel_open else [28, 0, 972],
    gap="small"
)

# ── TOGGLE BUTTON ─────────────────────────────────────────────────────────────
with toggle_col:
    icon = ICON_OPEN if st.session_state.panel_open else ICON_CLOSE
    if st.button(icon, key="panel_toggle"):
        st.session_state.panel_open = not st.session_state.panel_open
        st.rerun()

# ── LEFT PANEL ────────────────────────────────────────────────────────────────
if st.session_state.panel_open:
    with panel_col:
        st.markdown('<div class="panel-wrap">', unsafe_allow_html=True)

        # Header
        st.markdown("""
        <div class="panel-head">
          <div class="panel-logo-row">
            <span class="p-cross">✝</span>
            <div class="p-name">
              Bishop A.A Mayungbo
              <span>Ministry Library</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Search
        search_query = st.text_input("s", placeholder="🔍  Search…", label_visibility="collapsed")

        # Media
        if os.path.exists("media"):
            media_files = os.listdir("media")
            pdfs   = sorted([f for f in media_files if f.lower().endswith(".pdf")])
            audios = sorted([f for f in media_files if f.lower().endswith(".mp3")])
            if search_query:
                q      = search_query.lower()
                pdfs   = [f for f in pdfs   if q in f.lower()]
                audios = [f for f in audios if q in f.lower()]

            if pdfs:
                st.markdown('<span class="sec-lbl">E-Books</span>', unsafe_allow_html=True)
                for pdf in pdfs:
                    display = pdf.replace("_"," ").replace(".pdf","")
                    with st.expander(f"📄  {display}"):
                        pdf_viewer(f"media/{pdf}", width=230)
                        with open(f"media/{pdf}", "rb") as fh:
                            st.download_button("Download PDF", fh, file_name=pdf,
                                               mime="application/pdf", use_container_width=True)

            if audios:
                st.markdown('<span class="sec-lbl">Sermons</span>', unsafe_allow_html=True)
                for audio in audios:
                    display = audio.replace("_"," ").replace(".mp3","")
                    with st.expander(f"🎙  {display}"):
                        st.audio(f"media/{audio}")
                        with open(f"media/{audio}", "rb") as fh:
                            st.download_button("Download Audio", fh, file_name=audio,
                                               mime="audio/mpeg", use_container_width=True)

            if search_query and not pdfs and not audios:
                st.markdown(f'<p class="no-match">No results for "{search_query}"</p>', unsafe_allow_html=True)
            if not search_query and not pdfs and not audios:
                st.markdown('<p class="no-match" style="margin-top:.6rem;">Add PDFs or MP3s to the <strong>media/</strong> folder.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="no-match" style="margin-top:.6rem;">No <code>media/</code> folder found.</p>', unsafe_allow_html=True)

        st.divider()

        # Admin — hidden behind lock icon expander, no visible password hint
        with st.expander("🔒  Admin"):
            admin_pass = st.text_input("pw", placeholder="Password…", type="password", label_visibility="collapsed")
            if admin_pass == "bishop2024":
                if not st.session_state.admin_unlocked:
                    st.session_state.admin_unlocked = True
                st.success("Access granted.")
                admin_q = st.text_input("aq", placeholder="Member's question…", label_visibility="collapsed")
                admin_a = st.text_area("aa", placeholder="Bishop's answer…",    label_visibility="collapsed")
                if st.button("💾  Save to AI Memory"):
                    if admin_q and admin_a:
                        with st.spinner("Saving…"):
                            vectorstore.add_texts(
                                texts=[f"Question: {admin_q}\nAnswer: {admin_a}"],
                                metadatas=[{"source": "Bishop's Direct Answer"}]
                            )
                        st.success("Saved.")
                    else:
                        st.warning("Fill in both fields.")
            elif admin_pass:
                st.error("Incorrect password.")

        st.markdown('<p class="panel-foot">Powered by AI · Bishop A.A Mayungbo\'s Teachings</p>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── CHAT AREA ─────────────────────────────────────────────────────────────────
with chat_col:

    # Top bar — title only
    st.markdown("""
    <div class="chat-topbar">
      <span class="topbar-cross">✝</span>
      <span class="topbar-title">Bishop A.A Mayungbo Ministry AI</span>
    </div>
    """, unsafe_allow_html=True)

    # Welcome
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome">
          <span class="w-cross">✝</span>
          <div class="w-title">Calvary Greetings, Beloved</div>
          <div class="w-rule"></div>
          <p class="w-sub">
            Ask me anything about faith, the Bible,<br>
            or Bishop A.A Mayungbo's teachings.
          </p>
        </div>
        """, unsafe_allow_html=True)

    # Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if user_question := st.chat_input("Ask anything about faith or the teachings…"):
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
