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
st.set_page_config(page_title="Bishop A.A Mayungbo Ministry AI", page_icon="📖", layout="wide")
st.title("Welcome to the Ministry Chat 📖")
st.caption("Ask me anything about faith, the Bible, and spiritual growth.")

# ==========================================
# 3. DYNAMIC MEDIA LIBRARY & ADMIN SIDEBAR
# ==========================================
with st.sidebar:
    st.title("📚 Ministry Library")
    
    # --- MEDIA FILES ---
    if os.path.exists("media"):
        media_files = os.listdir("media")
        pdfs = [f for f in media_files if f.lower().endswith(".pdf")]
        audios = [f for f in media_files if f.lower().endswith(".mp3")]
        
        if pdfs:
            st.subheader("📖 E-Books")
            for pdf in pdfs:
                display_name = pdf.replace("_", " ").replace(".pdf", "")
                with st.expander(f"📄 {display_name}"):
                    pdf_viewer(f"media/{pdf}", width=300)
                    with open(f"media/{pdf}", "rb") as f:
                        st.download_button(label="📥 Download", data=f, file_name=pdf, mime="application/pdf", use_container_width=True)
        
        if audios:
            st.subheader("🎧 Sermons")
            for audio in audios:
                display_name = audio.replace("_", " ").replace(".mp3", "")
                with st.expander(f"🎙️ {display_name}"):
                    st.audio(f"media/{audio}")
                    with open(f"media/{audio}", "rb") as f:
                        st.download_button(label="📥 Download", data=f, file_name=audio, mime="audio/mpeg", use_container_width=True)
    
    st.divider()

    # --- ADMIN PANEL (Hidden behind a password) ---
    st.subheader("🔒 Ministry Admin")
    admin_pass = st.text_input("Enter Admin Password:", type="password")
    
    if admin_pass == "bishop2024": # You can change this password to anything you want
        st.success("Admin Access Granted!")
        
        st.markdown("**Teach the AI:**")
        admin_q = st.text_input("1. What was the member's question?")
        admin_a = st.text_area("2. What is the Bishop's answer?")
        
        if st.button("💾 Save to AI Memory"):
            if admin_q and admin_a:
                with st.spinner("Teaching the AI..."):
                    # Combine Q&A into a single text block
                    new_knowledge = f"Question: {admin_q}\nAnswer: {admin_a}"
                    # Upload to Pinecone
                    vectorstore.add_texts(
                        texts=[new_knowledge], 
                        metadatas=[{"source": "Bishop's Direct Answer (Telegram)"}]
                    )
                st.success("✅ The AI has learned this! It will never forget.")
            else:
                st.warning("Please fill in both the question and the answer.")
    else:
        if admin_pass:
            st.error("Incorrect Password.")

    st.divider()
    st.caption("Powered by AI | Trained on Bishop A.A Mayungbo's Teachings")

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
    
    message = f"🔔 *NEW QUESTION FOR THE BISHOP*\n\nA member asked a question that is not in the library:\n\n❓ *{question_text}*\n\nPlease reply via the Admin Panel on the website."
    
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {
        "chat_id": telegram_chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
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
        clean_name = os.path.basename(source_file).replace("_", " ").replace(".pdf", "").replace(".mp3", "").replace(".txt", "")
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("What is your question today?"):
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        with st.spinner("Searching the teachings..."):
            history_str = ""
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    history_str += f"User: {msg['content']}\n"
                else:
                    history_str += f"Assistant: {msg['content']}\n"
            
            response = rag_chain.invoke({
                "question": user_question,
                "chat_history": history_str
            })
            st.markdown(response)

            # --- THE MAGIC TRIGGER ---
            # If the AI says it doesn't know, we instantly alert the Bishop on Telegram!
            if "not currently in my library" in response:
                send_telegram_alert(user_question)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
