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
st.set_page_config(page_title="Bishop A.A Mayungbo Ministry AI", page_icon="📖")
st.title("Welcome to the Ministry Chat 📖")
st.caption("Ask me anything about faith, the Bible, and spiritual growth.")

# ==========================================
# 3. ADD THE DOWNLOAD & MEDIA SIDEBAR
# ==========================================
with st.sidebar:
    st.title("📚 Ministry Resources")
    
    # --- PDF SECTION ---
    st.subheader("📖 Read the Book")
    if os.path.exists("real_book.pdf"):
        # Show PDF Viewer directly in the sidebar
        pdf_viewer("real_book.pdf", width=300)
        # Show Download Button
        with open("real_book.pdf", "rb") as pdf_file:
            st.download_button(
                label="📥 Download PDF",
                data=pdf_file,
                file_name="Bishop_A.A_Mayungbo_Book.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("Book PDF not found in the repository.")
        
    st.divider()
    
    # --- AUDIO SECTION ---
    st.subheader("🎧 Listen to the Sermon")
    if os.path.exists("sermon_audio.mp3"):
        # Show native Streamlit Audio Player
        st.audio("sermon_audio.mp3")
        # Show Download Button
        with open("sermon_audio.mp3", "rb") as audio_file:
            st.download_button(
                label="📥 Download Audio",
                data=audio_file,
                file_name="Bishop_A.A_Mayungbo_Sermon.mp3",
                mime="audio/mpeg"
            )
    else:
        st.warning("Audio file not found in the repository.")
        
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
# 5. THE "BISHOP A.A MAYUNGBO" PERSONA & MEMORY PROMPT
# ==========================================
template = """
You are the official digital assistant and representative of Bishop A.A Mayungbo. 
Your tone is warm, deeply spiritual, encouraging, and filled with the grace of God. You frequently use biblical and charismatic terms of endearment such as "Beloved", "Calvary greetings", "Man/Woman of God", "Hallelujah", and "By his grace".

Your primary source of truth is the provided context, which consists exclusively of Bishop A.A Mayungbo's ebooks, manuscripts, and sermon transcripts. You must prioritize the teachings found in the ebooks above all else.

RULES:
1. ALWAYS answer using the provided context. 
2. When you greet the user or start a new topic, start your response with a warm spiritual greeting like "Calvary greetings, beloved!" or "Grace and peace to you."
3. Read the Chat History to understand the flow of conversation. If the user says "explain that more", look at the previous messages to know what "that" is.
4. If the answer is NOT in the context, you MUST reply exactly with: "Calvary greetings, beloved. That specific question is not currently in my library of Bishop A.A Mayungbo's teachings. I have noted it down and will present it to the ministry team for you!"
5. Never break character. You are not a generic AI; you are the digital voice of this ministry.

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
    return "\n\n".join(doc.page_content for doc in docs)

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
