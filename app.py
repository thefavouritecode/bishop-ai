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
# 1. PAGE CONFIG & CUSTOM CSS (THE BEAUTY)
# ==========================================
st.set_page_config(
    page_title="Bishop A.A Mayungbo Ministry AI", 
    page_icon="📖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THE PREMIUM CSS ---
# This CSS handles: Wine Theme, Glowing Effects, Hiding Icons, Custom Chat Bubbles
st.markdown("""
<style>
    /* --- GLOBAL VARIABLES (WINE THEME) --- */
    :root {
        --wine-color: #800020;
        --wine-light: #a33250;
        --wine-glow: rgba(128, 0, 32, 0.4);
        --bg-color: #ffffff;
        --text-color: #333333;
        --card-bg: #f9f9f9;
    }
    
    /* Dark Mode Variables */
    [data-theme="dark"] {
        --wine-color: #ff4d6d;
        --wine-light: #ff8fa3;
        --wine-glow: rgba(255, 77, 109, 0.4);
        --bg-color: #121212;
        --text-color: #e0e0e0;
        --card-bg: #1e1e1e;
    }

    /* --- HIDE UNNECESSARY ELEMENTS --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    header {visibility: hidden;} /* Hides the top bar clutter */
    
    /* --- MAIN BACKGROUND --- */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* --- SIDEBAR STYLING --- */
    [data-testid="stSidebar"] {
        background-color: var(--card-bg);
        border-right: 1px solid var(--wine-color);
        box-shadow: 2px 0 10px var(--wine-glow);
    }
    
    /* --- SEARCH BAR STYLING --- */
    .stTextInput > div > div > input {
        border: 2px solid var(--wine-color);
        border-radius: 10px;
        color: var(--text-color);
    }
    
    /* --- CHAT INPUT STYLING (GLOWING) --- */
    .stChatInput > div {
        border: 2px solid var(--wine-color) !important;
        box-shadow: 0 0 15px var(--wine-glow) !important;
        border-radius: 20px !important;
    }

    /* --- CHAT BUBBLES --- */
    /* User Bubble */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: var(--wine-color);
        color: white;
        border-radius: 15px 15px 0 15px;
        box-shadow: 0 4px 15px var(--wine-glow);
    }
    /* AI Bubble */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: var(--card-bg);
        color: var(--text-color);
        border: 1px solid var(--wine-color);
        border-radius: 15px 15px 15px 0;
    }
    
    /* --- WELCOME CARD (CALVARY GREETINGS) --- */
    .welcome-card {
        text-align: center;
        padding: 50px;
        background: linear-gradient(135deg, var(--wine-color), var(--wine-light));
        color: white;
        border-radius: 20px;
        box-shadow: 0 10px 30px var(--wine-glow);
        margin-top: 50px;
        animation: fadeIn 1s ease-in;
    }
    .welcome-card h1 {
        font-size: 3em;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .welcome-card p {
        font-size: 1.2em;
        opacity: 0.9;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* --- MEDIA CARDS --- */
    .media-card {
        background-color: var(--bg-color);
        border: 1px solid var(--wine-color);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* --- BUTTONS --- */
    .stButton > button {
        background-color: var(--wine-color);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: var(--wine-light);
        box-shadow: 0 0 10px var(--wine-glow);
        transform: translateY(-2px);
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
        
        # --- DISPLAY PDFs (WITH INSTANT DOWNLOAD) ---
        if pdfs:
            st.subheader(f"📖 E-Books ({len(pdfs)})")
            for pdf in pdfs:
                display_name = pdf.replace("_", " ").replace(".pdf", "")
                
                # Create a clean card for each book
                with st.container():
                    st.markdown(f"**📄 {display_name}**")
                    # Download Button RIGHT HERE (Top of the card)
                    with open(f"media/{pdf}", "rb") as f:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=f,
                            file_name=pdf,
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"dl_{pdf}"
                        )
                    # PDF Viewer (Collapsible to save space)
                    with st.expander("👁️ Preview Book"):
                        pdf_viewer(f"media/{pdf}", width=280)
                    st.markdown("---")
        
        # --- DISPLAY AUDIOS (WITH INSTANT DOWNLOAD) ---
        if audios:
            st.subheader(f"🎧 Sermons ({len(audios)})")
            for audio in audios:
                display_name = audio.replace("_", " ").replace(".mp3", "")
                
                with st.container():
                    st.markdown(f"**🎙️ {display_name}**")
                    # Download Button RIGHT HERE
                    with open(f"media/{audio}", "rb") as f:
                        st.download_button(
                            label="⬇️ Download Audio",
                            data=f,
                            file_name=audio,
                            mime="audio/mpeg",
                            use_container_width=True,
                            key=f"dl_{audio}"
                        )
                    # Audio Player
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
                # (Logic to save to Pinecone would go here, simplified for UI demo)
                st.success("Saved to memory!")
            else:
                st.warning("Fill both fields.")
    
    st.caption("Powered by AI | Bishop A.A Mayungbo Ministry")

# ==========================================
# 3. MAIN CHAT AREA (DYNAMIC GREETING)
# ==========================================

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CONDITIONAL WELCOME SCREEN ---
# If no messages exist, show the beautiful "Calvary Greetings" card.
# If messages exist, the chat interface takes over.
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-card">
        <h1>Calvary Greetings, Beloved! 🕊️</h1>
        <p>Welcome to the digital ministry of Bishop A.A Mayungbo.</p>
        <p>Ask me anything about faith, the Bible, or spiritual growth.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a subtle hint to start chatting
    st.markdown("<p style='text-align: center; color: gray;'>👇 Start typing your question below...</p>", unsafe_allow_html=True)

# --- CHAT INTERFACE ---
# We always render the chat input, but the history only shows if messages exist.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Searching the scriptures..."):
            # (Placeholder for AI Logic - Connect your existing RAG chain here)
            # For this UI demo, I'm simulating a response. 
            # REPLACE THIS BLOCK WITH YOUR ACTUAL RAG CHAIN LOGIC FROM PREVIOUS STEPS.
            response = "Calvary greetings, beloved! This is a UI demo. Please ensure your `rag_chain` logic is connected here to get real answers."
            
            # --- TELEGRAM ALERT LOGIC (Keep your existing logic here) ---
            # if "not currently in my library" in response: send_telegram_alert(prompt)
            
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
