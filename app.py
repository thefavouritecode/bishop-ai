import os
import logging
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter
from streamlit_pdf_viewer import pdf_viewer
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
# 3. DYNAMIC MEDIA LIBRARY SIDEBAR
# ==========================================
with st.sidebar:
    st.title("📚 Ministry Library")
    st.markdown("Read, listen, and download materials directly.")
    
    # Check if the media folder exists
    if os.path.exists("media"):
        media_files = os.listdir("media")
        pdfs = [f for f in media_files if f.lower().endswith(".pdf")]
        audios = [f for f in media_files if f.lower().endswith(".mp3")]
        
        # --- DISPLAY PDFs ---
        if pdfs:
            st.subheader("📖 E-Books & Manuscripts")
            for pdf in pdfs:
                # Clean up the name for display (remove underscores and .pdf)
                display_name = pdf.replace("_", " ").replace(".pdf", "").replace(".PDF", "")
                with st.expander(f"📄 {display_name}"):
                    pdf_viewer(f"media/{pdf}", width=300)
                    with open(f"media/{pdf}", "rb") as f:
                        st.download_button(
                            label="📥 Download PDF",
                            data=f,
                            file_name=pdf,
                            mime="application/pdf",
                            use_container_width=True
                        )
        
        # --- DISPLAY AUDIOS ---
        if audios:
            st.subheader("🎧 Sermons & Audio Messages")
            for audio in audios:
                display_name = audio.replace("_", " ").replace(".mp3", "").replace(".MP3", "")
                with st.expander(f"🎙️ {display_name}"):
                    st.audio(f"media/{audio}")
                    with open(f"media/{audio}", "rb") as f:
                        st.download_button(
                            label="📥 Download Audio",
                            data=f,
                            file_name=audio,
                            mime="audio/mpeg",
                            use_container_width=True
                        )
    else:
        st.warning("The 'media' folder was not found in the repository.")
        
    st.divider()
    st.caption("Powered by AI | Trained on Bishop A.A Mayungbo's Teachings")

# ==========================================
# 4. LOAD AI BRAIN & MEMORY
# ==========================================
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
index_name = "preacher-books"

@st.cache_resource
def load_ai_system():
    llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.4)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return llm, retriever

llm, retriever = load_ai_system()

# ==========================================
# 5. THE "BISHOP A.A MAYUNGBO" PERSONA & SUGGESTION PROMPT
# ==========================================
template = """
You are the official digital assistant and representative of Bishop A.A Mayungbo. 
Your tone is warm, deeply spiritual, encouraging, and filled with the grace of God. You frequently use biblical and charismatic terms of endearment such as "Beloved", "Calvary greetings", "Man/Woman of God", "Hallelujah", and "By his grace".

Your primary source of truth is the provided context, which consists exclusively of Bishop A.A Mayungbo's ebooks, manuscripts, and sermon transcripts. 

RULES:
1. ALWAYS answer using the provided context. 
2. When you greet the user or start a new topic, start your response with a warm spiritual greeting like "Calvary greetings, beloved!" or "Grace and peace to you."
3. Read the Chat History to understand the flow of conversation.
4. If the answer is NOT in the context, you MUST reply exactly with: "Calvary greetings, beloved. That specific question is not currently in my library of Bishop A.A Mayungbo's teachings. I have noted it down and will present it to the ministry team for you!"
5. NEVER break character. 
6. CRITICAL SUGGESTION RULE: Look at the "Excerpt from: [Name]" tags in the context. At the end of your answer, warmly suggest the user to read or listen to that specific material. Say something like: "For deeper insight on this, I highly recommend you read/listen to [Name] available in the sidebar library."

Chat History:
{chat_history}

Context from the ebooks and sermons:
{context}

User's Question: 
{question}

Helpful Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# This function now extracts the file name from the metadata and adds it to the text!
def format_docs(docs):
    formatted = []
    for doc in docs:
        # Langchain automatically saves the file path in metadata['source']
        source_file = doc.metadata.get("source", "Unknown Material")
        # Clean up the path to just the file name, remove underscores and extensions
        clean_name = os.path.basename(source_file).replace("_", " ").replace(".pdf", "").replace(".mp3", "").replace(".txt", "")
        
        formatted.append(f"--- Excerpt from: {clean_name} ---\n{doc.page_content}")
    return "\n\n".join(formatted)

# Build the RAG Chain with Memory
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
# 6. BUILD THE CHAT INTERFACE
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
    
    st.session_state.messages.append({"role": "assistant", "content": response})
