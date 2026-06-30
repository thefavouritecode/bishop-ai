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
    initial_sidebar_state="expanded", # Forces sidebar open on load
    menu_items={}
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── PREMIUM CSS (FIXED LAYOUT, WIDTH, & SIDEBAR) ──────────────────────────────
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

/* ── base ── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--txt) !important;
}

/* ── strip clutter ── */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], 
[data-testid="stStatusWidget"], .stDeployButton, a[href*="streamlit.io"] { 
  display: none !important; 
}

/* ════════════════════════════════════════
   SIDEBAR (FORCED OPEN & PROMINENT)
════════════════════════════════════════ */
[data-testid="stSidebar"] {
  display: block !important;
  width: 320px !important;
  min-width: 320px !important;
  max-width: 320px !important;
  background: var(--bg-sidebar) !important;
  border-right: 1px solid var(--sep) !important;
}

/* Giant Glowing Button if Sidebar is Collapsed on Mobile */
[data-testid="stSidebarCollapsedControl"] {
  top: 1rem !important; left: 1rem !important; z-index: 9999 !important;
}
[data-testid="stSidebarCollapsedControl"] button {
  background: var(--gold) !important;
  color: var(--bg) !important;
  width: 50px !important; height: 50px !important;
  border-radius: 50% !important;
  box-shadow: 0 0 20px var(--gold-glow) !important;
  border: none !important;
}

/* Sidebar Header */
.sb-head { padding: 1.5rem 1rem 1rem; border-bottom: 1px solid var(--sep); margin-bottom: 1rem; }
.sb-logo { display: flex; align-items: center; gap: 0.8rem; }
.sb-cross { font-size: 1.8rem; filter: drop-shadow(0 0 10px rgba(232,192,106,0.6)); animation: glow 3s ease-in-out infinite; }
@keyframes glow { 0%,100% { filter: drop-shadow(0 0 8px rgba(232,192,106,0.5)); } 50% { filter: drop-shadow(0 0 18px rgba(232,192,106,1)); } }
.sb-name { font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 700; color: var(--gold-lt); line-height: 1.2; }
.sb-name small { display: block; font-family: 'DM Sans', sans-serif; font-size: 0.7rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: var(--txt3); margin-top: 4px; }

/* Sidebar Elements */
[data-testid="stSidebar"] .stTextInput input, [data-testid="stSidebar"] .stTextArea textarea {
  background: var(--bg-input) !important; border: 1px solid var(--sep) !important; border-radius: var(--r) !important; color: var(--txt) !important;
}
[data-testid="stSidebar"] .stTextInput input:focus { border-color: var(--gold) !important; box-shadow: 0 0 0 2px rgba(232,192,106,0.15) !important; }

[data-testid="stSidebar"] button {
  background: transparent !important; border: 1px solid var(--sep) !important; color: var(--gold) !important;
  border-radius: var(--r) !important; transition: all 0.2s !important; font-weight: 500 !important;
}
[data-testid="stSidebar"] button:hover { background: rgba(232,192,106,0.1) !important; border-color: var(--gold) !important; }

.sec { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: var(--gold-dk); display: block; margin: 1rem 0 0.5rem; }

/* ════════════════════════════════════════
   MAIN CHAT AREA (PERFECT WIDTH & ALIGNMENT)
════════════════════════════════════════ */
.main .block-container { padding: 2rem 2rem 5rem 2rem !important; max-width: 1000px !important; margin: 0 auto !important; }

/* Top Bar */
.topbar { display: flex; align-items: center; gap: 0.8rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--sep); margin-bottom: 2rem; }
.tb-cross { font-size: 1.5rem; filter: drop-shadow(0 0 8px rgba(232,192,106,0.6)); }
.tb-title { font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 700; color: var(--gold-lt); }

/* Welcome Screen */
.welcome { text-align: center; padding: 4rem 1rem; max-width: 500px; margin: 0 auto; }
.wc { font-size: 3.5rem; display: block; animation: glow 3s ease-in-out infinite; margin-bottom: 1.5rem; }
.wt { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: var(--gold-lt); margin-bottom: 0.5rem; }
.wr { width: 60px; height: 2px; margin: 1rem auto; background: linear-gradient(90deg,transparent,var(--gold),transparent); }
.ws { font-size: 1rem; color: var(--txt2); line-height: 1.6; }

/* ── CHAT MESSAGES (FIXED WIDTH & ALIGNMENT) ── */
[data-testid="stChatMessage"] {
  padding: 0.25rem 0 !important;
  max-width: 100% !important;
}

/* The Bubble Wrapper */
[data-testid="stChatMessage"] > div:last-child {
  max-width: 80% !important;       /* Limits width to 80% */
  width: fit-content !important;   /* Shrinks to fit the text perfectly */
  display: table !important;       /* Ensures height wraps tightly around text */
  border-radius: 18px !important;
  padding: 0.8rem 1.2rem !important;
  box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
}

/* User Message (Pushed strictly to the RIGHT) */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:last-child {
  margin-left: auto !important;
  margin-right: 0 !important;
  background: linear-gradient(135deg, #1a1c29, #252836) !important;
  border: 1px solid rgba(108,99,255,0.3) !important;
  color: #fff !important;
}

/* Assistant Message (Pushed strictly to the LEFT) */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div:last-child {
  margin-left: 0 !important;
  margin-right: auto !important;
  background: var(--bg-card) !important;
  border: 1px solid var(--sep) !important;
  color: var(--txt) !important;
}

/* Ensure text inside bubbles is left-aligned */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { text-align: left !important; width: 100% !important; }

/* ── CHAT INPUT (GLOWING PILL) ── */
[data-testid="stChatInputContainer"] { padding: 1rem 0 !important; background: transparent !important; border: none !important; }
[data-testid="stChatInputContainer"] > div {
  background: var(--bg-input) !important; border: 1.5px solid rgba(232,192,106,0.3) !important;
  border-radius: var(--pill) !important; box-shadow: 0 0 20px rgba(232,192,106,0.1) !important; transition: all 0.3s !important;
}
[data-testid="stChatInputContainer"] > div:focus-within { border-color: var(--gold) !important; box-shadow: 0 0 30px rgba(232,192,106,0.25) !important; }
[data-testid="stChatInputContainer"] textarea { color: var(--txt) !important; font-family: 'DM Sans', sans-serif !important; }
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
    if not telegram_token or not telegram_chat_id: return
    try:
        requests.post(f"https://api.telegram.org/bot{telegram_token}/sendMessage",
            data={"chat_id": telegram_chat_id, "text": f"🔔 *NEW QUESTION*\n\n❓ {q}", "parse_mode": "Markdown"})
    except: pass

# ── PROMPT & CHAIN ────────────────────────────────────────────────────────────
template = """
You are the digital assistant of Bishop A.A Mayungbo.
Tone: Warm, spiritual, authoritative. Use "Beloved", "Calvary greetings".
RULES:
1. Answer ONLY from context.
2. Start with "Calvary greetings, beloved!"
3. If NOT in context, reply EXACTLY: "Calvary greetings, beloved. That specific question is not currently in my library of Bishop A.A Mayungbo's teachings. I have noted it down and will present it to the ministry team for you!"
4. Suggest reading the source material in the library.

Chat History: {chat_history}
Context: {context}
Question: {question}
Answer:
"""
prompt_tmpl = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    out = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        name = os.path.basename(src).replace("_"," ").replace(".pdf","").replace(".mp3","")
        out.append(f"--- Excerpt from: {name} ---\n{doc.page_content}")
    return "\n\n".join(out)

rag_chain = (
    {"context": itemgetter("question") | retriever | format_docs, "question": itemgetter("question"), "chat_history": itemgetter("chat_history")}
    | prompt_tmpl | llm | StrOutputParser()
)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-head">
      <div class="sb-logo">
        <span class="sb-cross">✝</span>
        <div class="sb-name">Bishop A.A Mayungbo<small>Ministry Library</small></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    search = st.text_input("_s", placeholder="🔍  Search books & sermons…", label_visibility="collapsed")

    if os.path.exists("media"):
        all_files = os.listdir("media")
        pdfs   = sorted([f for f in all_files if f.lower().endswith(".pdf")])
        audios = sorted([f for f in all_files if f.lower().endswith(".mp3")])
        
        if search:
            q = search.lower()
            pdfs = [f for f in pdfs if q in f.lower()]
            audios = [f for f in audios if q in f.lower()]

        if pdfs:
            st.markdown('<span class="sec">📖 E-Books</span>', unsafe_allow_html=True)
            for pdf in pdfs:
                label = pdf.replace("_", " ").replace(".pdf", "")
                st.markdown(f"**📄 {label}**")
                
                # Create two columns for View and Download
                col_view, col_dl = st.columns(2)
                
                with col_dl:
                    with open(f"media/{pdf}", "rb") as fh:
                        st.download_button("⬇️ Download", fh, file_name=pdf, mime="application/pdf", use_container_width=True, key=f"dl_{pdf}")
                
                with col_view:
                    if st.button("👁️ Preview", key=f"prev_{pdf}", use_container_width=True):
                        st.session_state[f"view_{pdf}"] = not st.session_state.get(f"view_{pdf}", False)

                # If preview is toggled on, show the viewer below the buttons
                if st.session_state.get(f"view_{pdf}", False):
                    pdf_viewer(f"media/{pdf}", width=280)
                
                st.markdown("---")

        if audios:
            st.markdown('<span class="sec">🎧 Sermons</span>', unsafe_allow_html=True)
            for audio in audios:
                label = audio.replace("_", " ").replace(".mp3", "")
                st.markdown(f"**🎙️ {label}**")
                
                col_view, col_dl = st.columns(2)
                
                with col_dl:
                    with open(f"media/{audio}", "rb") as fh:
                        st.download_button("⬇️ Download", fh, file_name=audio, mime="audio/mpeg", use_container_width=True, key=f"dl_{audio}")
                
                with col_view:
                    if st.button("▶️ Listen", key=f"prev_{audio}", use_container_width=True):
                        st.session_state[f"view_{audio}"] = not st.session_state.get(f"view_{audio}", False)

                if st.session_state.get(f"view_{audio}", False):
                    st.audio(f"media/{audio}")
                
                st.markdown("---")
    else:
        st.info("Add a 'media' folder to your repository.")

    st.divider()

    with st.expander("🔒  Ministry Admin"):
        pw = st.text_input("_pw", placeholder="Password…", type="password", label_visibility="collapsed")
        if pw == "bishop2024":
            st.success("Access granted.")
            aq = st.text_input("_aq", placeholder="Member's question…", label_visibility="collapsed")
            aa = st.text_area("_aa", placeholder="Bishop's answer…", label_visibility="collapsed")
            if st.button("💾  Teach AI"):
                if aq and aa:
                    vectorstore.add_texts(texts=[f"Question: {aq}\nAnswer: {aa}"], metadatas=[{"source": "Admin Teach"}])
                    st.success("Saved to memory.")
        elif pw: st.error("Incorrect password.")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT
# ══════════════════════════════════════════════════════════════════════════════

# Top Bar
st.markdown("""
<div class="topbar">
  <span class="tb-cross">✝</span>
  <span class="tb-title">Bishop A.A Mayungbo Ministry AI</span>
</div>
""", unsafe_allow_html=True)

# Welcome Screen
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
      <span class="wc">✝</span>
      <div class="wt">Calvary Greetings, Beloved</div>
      <div class="wr"></div>
      <p class="ws">Ask me anything about faith, the Bible,<br>or Bishop A.A Mayungbo's teachings.</p>
    </div>
    """, unsafe_allow_html=True)

# Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if question := st.chat_input("Ask anything about faith or the teachings…"):
    with st.chat_message("user"): st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching the teachings…"):
            history = "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in st.session_state.messages[:-1]])
            response = rag_chain.invoke({"question": question, "chat_history": history})
            st.markdown(response)
            if "not currently in my library" in response: send_telegram_alert(question)

    st.session_state.messages.append({"role": "assistant", "content": response})
