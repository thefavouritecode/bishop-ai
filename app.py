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

# ── SILENCE WARNINGS ─────────────────────────────────────────────────────────
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bishop A.A Mayungbo Ministry AI",
    page_icon="✝️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=Playfair+Display:wght@600;700&display=swap');

/* ── tokens ── */
:root {
  --bg:          #090b11;
  --bg-sidebar:  #0d0f18;
  --bg-card:     #12151e;
  --bg-input:    #161924;
  --gold:        #e8c06a;
  --gold-lt:     #f5dfa0;
  --gold-dk:     #7a5e28;
  --gold-glow:   rgba(232,192,106,0.15);
  --purple:      #6c63ff;
  --blue:        #4fa3e0;
  --txt:         #edeae2;
  --txt2:        #9a9284;
  --txt3:        #4e4c46;
  --sep:         rgba(232,192,106,0.10);
  --sep-h:       rgba(232,192,106,0.28);
  --r:           10px;
  --pill:        999px;
}

/* ── reset ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--txt) !important;
  height: 100%;
}

/* ── strip built-in chrome except sidebar controls ── */
#MainMenu,
footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton,
a[href*="streamlit.io"],
.viewerBadge_container__1QSob { display: none !important; }

/* ── main block container: no padding, full width ── */
.main .block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── header row that streamlit inserts above chat: remove gap ── */
[data-testid="stMainBlockContainer"] { padding-top: 0 !important; }

/* ════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: var(--bg-sidebar) !important;
  border-right: 1px solid var(--sep) !important;
  min-width: 270px !important;
  max-width: 270px !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 0 !important;
  display: flex;
  flex-direction: column;
  height: 100vh;
}

/* sidebar collapse arrow button ── keep it, just style it */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
  display: flex !important;
}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button {
  background: var(--bg-card) !important;
  border: 1px solid var(--sep) !important;
  color: var(--gold) !important;
  border-radius: 8px !important;
}
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover {
  border-color: var(--gold) !important;
  background: rgba(232,192,106,0.08) !important;
}

/* ── sidebar internal padding wrapper ── */
.sb-inner {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

/* ── sidebar header ── */
.sb-head {
  padding: 1.2rem 1.1rem 1rem;
  border-bottom: 1px solid var(--sep);
  flex-shrink: 0;
}
.sb-logo {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.sb-cross {
  font-size: 1.5rem;
  filter: drop-shadow(0 0 9px rgba(232,192,106,0.7));
  animation: glow 3s ease-in-out infinite;
  flex-shrink: 0;
  line-height: 1;
}
@keyframes glow {
  0%,100% { filter: drop-shadow(0 0 7px rgba(232,192,106,0.5)); }
  50%      { filter: drop-shadow(0 0 18px rgba(232,192,106,1.0)); }
}
.sb-name {
  font-family: 'Playfair Display', serif;
  font-size: 0.9rem;
  font-weight: 700;
  color: var(--gold-lt);
  line-height: 1.2;
  text-shadow: 0 0 18px rgba(232,192,106,0.25);
}
.sb-name small {
  display: block;
  font-family: 'DM Sans', sans-serif;
  font-size: 0.58rem;
  font-weight: 500;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--txt3);
  margin-top: 3px;
}

/* ── sidebar scrollable body ── */
.sb-body {
  flex: 1;
  overflow-y: auto;
  padding: 0.75rem 1rem 0.5rem;
  scrollbar-width: thin;
  scrollbar-color: var(--gold-dk) transparent;
}
.sb-body::-webkit-scrollbar { width: 2px; }
.sb-body::-webkit-scrollbar-thumb { background: var(--gold-dk); border-radius: 2px; }

/* ── sidebar footer ── */
.sb-foot {
  flex-shrink: 0;
  border-top: 1px solid var(--sep);
  padding: 0.65rem 1rem;
  font-size: 0.6rem;
  color: var(--txt3);
  text-align: center;
  letter-spacing: 0.04em;
}

/* ── section label ── */
.sec {
  font-size: 0.58rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--gold-dk);
  display: block;
  padding: 0.9rem 0 0.4rem;
}

/* ── no-results ── */
.nores {
  font-size: 0.72rem;
  color: var(--txt3);
  text-align: center;
  padding: 0.6rem 0.8rem;
  background: var(--bg-card);
  border: 1px dashed var(--sep);
  border-radius: var(--r);
  margin-top: 0.4rem;
}

/* ── expanders ── */
[data-testid="stSidebar"] [data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--sep) !important;
  border-radius: var(--r) !important;
  margin-bottom: 0.35rem !important;
  overflow: hidden;
  transition: border-color 0.2s;
}
[data-testid="stSidebar"] [data-testid="stExpander"]:hover {
  border-color: var(--sep-h) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  font-size: 0.77rem !important;
  font-weight: 500 !important;
  color: var(--txt) !important;
  padding: 0.5rem 0.75rem !important;
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
  color: var(--gold-lt) !important;
}

/* ── sidebar text inputs ── */
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] [data-testid="stTextArea"] textarea {
  background: var(--bg-input) !important;
  border: 1px solid var(--sep) !important;
  border-radius: var(--r) !important;
  color: var(--txt) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.8rem !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus,
[data-testid="stSidebar"] [data-testid="stTextArea"] textarea:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 2px rgba(232,192,106,0.1) !important;
  outline: none !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder,
[data-testid="stSidebar"] [data-testid="stTextArea"] textarea::placeholder {
  color: var(--txt3) !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] label,
[data-testid="stSidebar"] [data-testid="stTextArea"] label {
  display: none !important;
}

/* ── sidebar buttons ── */
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button,
[data-testid="stSidebar"] [data-testid="stButton"] button {
  background: transparent !important;
  border: 1px solid var(--sep) !important;
  border-radius: var(--r) !important;
  color: var(--gold) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.73rem !important;
  font-weight: 500 !important;
  width: 100% !important;
  transition: all 0.18s !important;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover,
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
  background: rgba(232,192,106,0.07) !important;
  border-color: var(--gold) !important;
}

/* ── sidebar alerts ── */
[data-testid="stSidebar"] [data-testid="stAlert"] {
  border-radius: var(--r) !important;
  font-size: 0.73rem !important;
  font-family: 'DM Sans', sans-serif !important;
  padding: 0.4rem 0.7rem !important;
}
[data-testid="stSidebar"] .stSuccess { background: rgba(61,220,132,0.08) !important; border-color: rgba(61,220,132,0.2) !important; color: #7ed9a8 !important; }
[data-testid="stSidebar"] .stError   { background: rgba(220,60,60,0.08)   !important; border-color: rgba(220,60,60,0.2)   !important; color: #f09090  !important; }
[data-testid="stSidebar"] .stWarning { background: rgba(232,192,106,0.08) !important; border-color: rgba(232,192,106,0.2) !important; color: var(--gold-lt) !important; }

/* ── sidebar divider ── */
[data-testid="stSidebar"] hr,
[data-testid="stSidebar"] [data-testid="stDivider"] {
  border-color: var(--sep) !important;
  margin: 0.6rem 0 !important;
}

/* ── audio in sidebar ── */
[data-testid="stSidebar"] audio {
  width: 100% !important;
  border-radius: var(--r) !important;
  filter: invert(0.75) hue-rotate(185deg) brightness(0.9);
  margin: 0.4rem 0 !important;
}

/* ════════════════════════════════════════
   MAIN CHAT AREA
════════════════════════════════════════ */

/* topbar */
.topbar {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  padding: 0.8rem 1.6rem;
  background: var(--bg-sidebar);
  border-bottom: 1px solid var(--sep);
}
.tb-cross {
  font-size: 1.05rem;
  filter: drop-shadow(0 0 7px rgba(232,192,106,0.65));
  animation: glow 3s ease-in-out infinite;
}
.tb-title {
  font-family: 'Playfair Display', serif;
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--gold-lt);
  text-shadow: 0 0 14px rgba(232,192,106,0.22);
}

/* welcome screen */
.welcome {
  text-align: center;
  padding: 4rem 1rem 2rem;
  max-width: 420px;
  margin: 0 auto;
}
.wc { font-size: 2.8rem; display: block; animation: glow 3s ease-in-out infinite; margin-bottom: 1rem; }
.wt {
  font-family: 'Playfair Display', serif;
  font-size: clamp(1.3rem, 2.5vw, 1.75rem);
  font-weight: 700;
  color: var(--gold-lt);
  text-shadow: 0 0 30px rgba(232,192,106,0.3);
  line-height: 1.3;
  margin-bottom: 0.5rem;
}
.wr { width: 48px; height: 1px; margin: 0.8rem auto; background: linear-gradient(90deg,transparent,var(--gold),transparent); }
.ws { font-size: 0.85rem; color: var(--txt2); line-height: 1.65; }

/* ── chat messages ── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0.18rem 0 !important;
  max-width: 700px;
  margin-left: auto !important;
  margin-right: auto !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
  font-size: 0.875rem !important;
  line-height: 1.68 !important;
  color: var(--txt) !important;
  margin: 0 !important;
}
[data-testid="stChatMessageAvatarUser"] {
  background: linear-gradient(135deg, var(--purple), var(--blue)) !important;
  border-radius: 50% !important;
  width: 28px !important; height: 28px !important; min-width: 28px !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
  background: linear-gradient(135deg, var(--gold-dk), var(--gold)) !important;
  border-radius: 50% !important;
  width: 28px !important; height: 28px !important; min-width: 28px !important;
}
[data-testid="stChatMessage"] > div:last-child {
  background: var(--bg-card) !important;
  border: 1px solid var(--sep) !important;
  border-radius: var(--r) !important;
  padding: 0.6rem 0.95rem !important;
  box-shadow: 0 1px 8px rgba(0,0,0,0.2) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:last-child {
  background: #12152b !important;
  border-color: rgba(108,99,255,0.18) !important;
}

/* ── chat input: pill + glow ── */
[data-testid="stChatInputContainer"] {
  padding: 0.75rem 2rem 1rem !important;
  background: var(--bg) !important;
  border-top: 1px solid var(--sep) !important;
  display: flex !important;
  justify-content: center !important;
}
/* the inner wrapper that holds textarea + button */
[data-testid="stChatInputContainer"] > div {
  background: var(--bg-input) !important;
  border: 1.5px solid rgba(232,192,106,0.32) !important;
  border-radius: var(--pill) !important;
  max-width: 660px !important;
  width: 100% !important;
  box-shadow:
    0 0 0 4px rgba(232,192,106,0.055),
    0 0 20px rgba(232,192,106,0.10),
    0 0 50px rgba(232,192,106,0.05) !important;
  transition: border-color 0.25s, box-shadow 0.25s !important;
  overflow: hidden;
}
[data-testid="stChatInputContainer"] > div:focus-within {
  border-color: rgba(232,192,106,0.7) !important;
  box-shadow:
    0 0 0 4px rgba(232,192,106,0.10),
    0 0 28px rgba(232,192,106,0.20),
    0 0 60px rgba(232,192,106,0.09) !important;
}
[data-testid="stChatInputContainer"] textarea {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  color: var(--txt) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.875rem !important;
  padding: 0.65rem 0.2rem 0.65rem 1.1rem !important;
  resize: none !important;
}
[data-testid="stChatInputContainer"] textarea::placeholder { color: var(--txt3) !important; }
[data-testid="stChatInputSubmitButton"] button {
  background: linear-gradient(135deg, #5a3e10, var(--gold)) !important;
  border: none !important;
  border-radius: 50% !important;
  width: 32px !important; height: 32px !important;
  margin: 5px 7px 5px 0 !important;
  color: #090b11 !important;
  transition: filter 0.15s, transform 0.15s !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
  filter: brightness(1.18) !important;
  transform: scale(1.07) !important;
}

/* ── spinner ── */
[data-testid="stSpinner"] > div { border-top-color: var(--gold) !important; }

/* ── scrollbars ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--gold-dk); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── LOAD AI ───────────────────────────────────────────────────────────────────
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

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
def send_telegram_alert(q):
    if not telegram_token or not telegram_chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{telegram_token}/sendMessage",
            data={
                "chat_id": telegram_chat_id,
                "text": f"🔔 *NEW QUESTION FOR THE BISHOP*\n\n❓ *{q}*\n\nPlease reply via the Admin Panel.",
                "parse_mode": "Markdown"
            }
        )
    except Exception as e:
        print(f"Telegram error: {e}")

# ── PROMPT & CHAIN ────────────────────────────────────────────────────────────
template = """
You are the official digital assistant and representative of Bishop A.A Mayungbo.
Your tone is warm, deeply spiritual, encouraging, and filled with the grace of God.
You use terms like "Beloved", "Calvary greetings", "Man/Woman of God", "Hallelujah", "By his grace".

RULES:
1. ALWAYS answer using the provided context.
2. Start with a warm spiritual greeting like "Calvary greetings, beloved!" or "Grace and peace to you."
3. Read Chat History to understand the conversation.
4. If the answer is NOT in the context, reply EXACTLY:
   "Calvary greetings, beloved. That specific question is not currently in my library of
   Bishop A.A Mayungbo's teachings. I have noted it down and will present it to the ministry team for you!"
5. NEVER break character.
6. When you find the answer, suggest reading the source material in the library panel.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
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

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Header
    st.markdown("""
    <div class="sb-head">
      <div class="sb-logo">
        <span class="sb-cross">✝</span>
        <div class="sb-name">
          Bishop A.A Mayungbo
          <small>Ministry Library</small>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Search
    search = st.text_input("_s", placeholder="🔍  Search books & sermons…", label_visibility="collapsed")

    # Media files
    if os.path.exists("media"):
        all_files = os.listdir("media")
        pdfs      = sorted([f for f in all_files if f.lower().endswith(".pdf")])
        audios    = sorted([f for f in all_files if f.lower().endswith(".mp3")])
        if search:
            q      = search.lower()
            pdfs   = [f for f in pdfs   if q in f.lower()]
            audios = [f for f in audios if q in f.lower()]

        if pdfs:
            st.markdown('<span class="sec">E-Books</span>', unsafe_allow_html=True)
            for pdf in pdfs:
                label = pdf.replace("_", " ").replace(".pdf", "")
                with st.expander(f"📄  {label}"):
                    pdf_viewer(f"media/{pdf}", width=220)
                    with open(f"media/{pdf}", "rb") as fh:
                        st.download_button("⬇  Download PDF", fh, file_name=pdf,
                                           mime="application/pdf", use_container_width=True)

        if audios:
            st.markdown('<span class="sec">Sermons</span>', unsafe_allow_html=True)
            for audio in audios:
                label = audio.replace("_", " ").replace(".mp3", "")
                with st.expander(f"🎙  {label}"):
                    st.audio(f"media/{audio}")
                    with open(f"media/{audio}", "rb") as fh:
                        st.download_button("⬇  Download Audio", fh, file_name=audio,
                                           mime="audio/mpeg", use_container_width=True)

        if search and not pdfs and not audios:
            st.markdown(f'<p class="nores">No results for "{search}"</p>', unsafe_allow_html=True)
        if not search and not pdfs and not audios:
            st.markdown('<p class="nores" style="margin-top:.5rem;">Add PDFs or MP3s to the <b>media/</b> folder.</p>',
                        unsafe_allow_html=True)
    else:
        st.markdown('<p class="nores" style="margin-top:.5rem;">No <code>media/</code> folder found.</p>',
                    unsafe_allow_html=True)

    st.divider()

    # Admin (hidden inside expander)
    with st.expander("🔒  Admin"):
        pw = st.text_input("_pw", placeholder="Enter password…", type="password", label_visibility="collapsed")
        if pw == "bishop2024":
            st.success("Access granted.")
            aq = st.text_input("_aq", placeholder="Member's question…", label_visibility="collapsed")
            aa = st.text_area("_aa", placeholder="Bishop's answer…",    label_visibility="collapsed")
            if st.button("💾  Save to AI Memory"):
                if aq and aa:
                    with st.spinner("Saving…"):
                        vectorstore.add_texts(
                            texts=[f"Question: {aq}\nAnswer: {aa}"],
                            metadatas=[{"source": "Bishop's Direct Answer"}]
                        )
                    st.success("Saved to memory.")
                else:
                    st.warning("Fill in both fields.")
        elif pw:
            st.error("Incorrect password.")

    st.markdown('<p class="sb-foot">Powered by AI · Bishop A.A Mayungbo\'s Teachings</p>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT
# ══════════════════════════════════════════════════════════════════════════════

# Top bar
st.markdown("""
<div class="topbar">
  <span class="tb-cross">✝</span>
  <span class="tb-title">Bishop A.A Mayungbo Ministry AI</span>
</div>
""", unsafe_allow_html=True)

# Welcome screen
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
      <span class="wc">✝</span>
      <div class="wt">Calvary Greetings, Beloved</div>
      <div class="wr"></div>
      <p class="ws">
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
if question := st.chat_input("Ask anything about faith or the teachings…"):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching the teachings…"):
            history = ""
            for m in st.session_state.messages[:-1]:
                history += f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}\n"

            response = rag_chain.invoke({"question": question, "chat_history": history})
            st.markdown(response)

            if "not currently in my library" in response:
                send_telegram_alert(question)

    st.session_state.messages.append({"role": "assistant", "content": response})
