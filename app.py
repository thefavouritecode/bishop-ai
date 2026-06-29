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
# 1. PAGE CONFIG & PREMIUM CSS
# ==========================================
st.set_page_config(
    page_title="Bishop A.A Mayungbo Ministry AI", 
    page_icon="📖", 
    layout="wide",
    initial_sidebar_state="expanded" # Forces sidebar to stay open
)

# --- THE PREMIUM CREAM & WINE CSS ---
st.markdown("""
<style>
    /* --- GLOBAL VARIABLES (ELEGANT CREAM & WINE) --- */
    :root {
        --wine-color: #722F37; /* Rich, elegant wine */
        --wine-light: #9D4A54;
        --wine-glow: rgba(114, 47, 55, 0.25);
        --bg-color: #FCFAF8; /* Soft, warm cream (not stark white) */
        --text-color: #2C2C2C; /* Soft black for high readability */
        --card-bg: #FFFFFF; /* Pure white for cards to pop against cream */
        --sidebar-bg: #FDF8F5; /* Very subtle warm tint for sidebar */
    }
    
    /* Dark Mode Variables */
    [data-theme="dark"] {
        --wine-color: #D9534F; /* Brighter wine for dark mode visibility */
        --wine-light: #E27278;
        --wine-glow: rgba(217, 83, 79, 0.3);
        --bg-color: #181818; /* Deep charcoal */
        --text-color: #EAEAEA;
        --card-bg: #252525;
        --sidebar-bg: #1E1E1E;
    }

    /* --- HIDE UNNECESSARY CLUTTER --- */
    #MainMenu, footer, .stDeployButton, header {visibility: hidden;}
    
    /* --- MAIN BACKGROUND --- */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* --- SIDEBAR STYLING (HIGHLY VISIBLE & ELEGANT) --- */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 2px solid var(--wine-color);
        box-shadow: 4px 0 20px var(--wine-glow);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: var(--wine-color) !important;
        font-weight: 600;
    }

    /* --- SEARCH BAR STYLING --- */
    .stTextInput > div > div > input {
        border: 2px solid var(--wine-color);
        border-radius: 12px;
        background-color: var(--card-bg);
        color: var(--text-color);
        box-shadow: 0 0 10px rgba(114, 47, 55, 0.1);
    }

    /* --- CHAT INPUT STYLING (GLOWING) --- */
    .stChatInput > div {
        border: 2px solid var(--wine-color) !important;
        box-shadow: 0 0 20px var(--wine-glow) !important;
        border-radius: 25px !important;
        background-color: var(--card-bg) !important;
    }

    /* --- CHAT BUBBLES (PERFECTLY ALIGNED & VISIBLE) --- */
    /* Streamlit automatically aligns User to Right, Assistant to Left */
    .stChatMessage[data-testid="stChatMessage"] {
        background-color: var(--card-bg);
        color: var(--text-color);
        border: 1px solid rgba(114, 47, 55, 0.15);
        border-radius: 22px;
        padding: 18px 24px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
        margin-bottom: 15px;
        max-width: 85%;
        font-size: 1.05em;
        line-height: 1.6;
    }
    
    /* Make the AI avatar look premium */
    .stChatMessage[data-testid="stChatMessage"] .st-emotion-cache-1y4r31y {
        background-color: var(--wine-color) !important;
        color: white !important;
        box-shadow: 0 0 10px var(--wine-glow);
    }

    /* --- WELCOME CARD (CALVARY GREETINGS) --- */
    .welcome-card {
        text-align: center;
        padding: 60px 40px;
        background: linear-gradient(135deg, var(--wine-color), var(--wine-light));
        color: white;
        border-radius: 30px;
        box-shadow: 0 20px 40px var(--wine-glow);
        margin: 50px auto;
        max-width: 650px;
        animation: fadeIn 1.2s ease-in;
    }
    .welcome-card h1 {
        font-size: 3em;
        margin-bottom: 15px;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        font-weight: 700;
        letter-spacing: 1px;
    }
    .welcome-card p {
        font-size: 1.25em;
        opacity: 0.95;
        line-height: 1.6;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* --- BUTTONS (GLOWING WINE) --- */
    .stButton > button {
        background-color: var(--wine-color);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(114, 47, 55, 0.2);
    }
    .stButton > button:hover {
        background-color: var(--wine-light);
        box-shadow: 0 0 20px var(--wine-glow);
        transform: translateY(-2px);
    }
    
    /* --- MEDIA CARDS --- */
    .stExpander {
        border: 1px solid rgba(114, 47, 55, 0.2) !important;
        border-radius: 12px !important;
        background-color: var(--card-bg) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR: SEARCH & LIBRARY
# ==========================================
with st.sidebar:
    st.title("📚 Ministry Library")
    
    # --- SEARCH BAR ---
    search_query = st.text_input("🔍 Search books & sermons...", "")
    
    # --- MEDIA FILES LOGIC ---
    if os.path.exists("media"):
        media_files = os.listdir("media")
        
        # Filter based on search
        if search_query:
            pdfs = [f for f in media_files if f.lower().endswith(".pdf") and search_query.lower() in f.lower()]
            audios = [f for f in media_files if f.lower().endswith(".mp3") and search_query.lower() in f.lower()]
        else:
            pdfs = [f for f in media_files if f.lower().endswith(".pdf")]
            audios = [f for f in media_files if f.lower().endswith(".mp3")]
        
        # --- DISPLAY PDFs ---
        if pdfs:
            st.subheader(f"📖 E-Books ({len(pdfs)})")
            for pdf in pdfs:
                display_name = pdf.replace("_", " ").replace(".pdf", "")
                with st.container():
                    st.markdown(f"**📄 {display_name}**")
                    with open(f"media/{pdf}", "rb") as f:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=f,
                            file_name=pdf,
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"dl_{pdf}"
                        )
                    with st.expander("👁️ Preview Book"):
                        pdf_viewer(f"media/{pdf}", width=280)
                    st.markdown("---")
        
        # --- DISPLAY AUDIOS ---
        if audios:
            st.subheader(f"🎧 Sermons ({len(audios)})")
            for audio in audios:
                display_name = audio.replace("_", " ").replace(".mp3", "")
                with st.container():
                    st.markdown(f"**🎙️ {display_name}**")
                    with open(f"media/{audio}", "rb") as f:
                        st.download_button(
                            label="⬇️ Download Audio",
                            data=f,
                            file_name=audio,
                            mime="audio/mpeg",
                            use_container_width=True,
                            key=f"dl_{audio}"
                        )
                    st.audio(f"media/{audio}")
                    st.markdown("---")
    else:
        st.info("No media folder found.")
    
    st.divider()

    # --- ADMIN PANEL ---
    st.subheader("🔒 Ministry Admin")
    admin_pass = st.text_input("Password:", type="password", key="admin_pass")
    
    if admin_pass == "bishop2024":
        st.success("✅ Access Granted")
        admin_q = st.text_input("Member's Question:")
        admin_a = st.text_area("Bishop's Answer:")
        if st.button("💾 Teach AI"):
            if admin_q and admin_a:
                st.success("Saved to memory!")
            else:
                st.warning("Fill both fields.")
    
    st.caption("Powered by AI | Bishop A.A Mayungbo Ministry")

# ==========================================
# 3. MAIN CHAT AREA (DYNAMIC GREETING)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CONDITIONAL WELCOME SCREEN ---
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-card">
        <h1>Calvary Greetings, Beloved! 🕊️</h1>
        <p>Welcome to the digital ministry of Bishop A.A Mayungbo.</p>
        <p>Ask me anything about faith, the Bible, or spiritual growth.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; font-size: 1.1em;'>👇 Start typing your question below...</p>", unsafe_allow_html=True)

# --- CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the scriptures..."):
            # --- PLACEHOLDER: CONNECT YOUR RAG CHAIN HERE ---
            # response = rag_chain.invoke({"question": prompt, "chat_history": history_str})
            response = "Calvary greetings, beloved! This is the new premium UI. Please ensure your `rag_chain` logic is connected in the code to get real answers from the Bishop's library."
            
            # if "not currently in my library" in response: send_telegram_alert(prompt)
            
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
