import os 
import time 
import tempfile # To store uploaded PDFs on disk temporarily

import streamlit as st
from dotenv import load_dotenv

# Langchain core classes & utilities
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

## Langchain LLM and chaining utilities
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# Text splitting & embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# vector store
from langchain_chroma import Chroma

## PDF file loader (loads a single PDF into docs)
from langchain_community.document_loaders import PyPDFLoader


# Loading environment variables ( HF_TOKEN, GROQ_API_KEY)
load_dotenv()


# Streamlit page setup
st.set_page_config(
    page_title = "📄 RAG Q&A with PDF & Chat History",
    layout = "wide",
    initial_sidebar_state= "expanded"
)
st.title("📄 RAG Q&A with PDF uploads and Chat History")

# Check and display SQLite version
import sqlite3
conn = sqlite3.connect(":memory:")
cursor = conn.cursor()
cursor.execute("select sqlite_version();")
version = cursor.fetchone()[0]
conn.close()
st.write(f"🔢 SQLite version: {version}")


st.sidebar.header("⚙️ Configuration")
st.sidebar.write(
    " - Enter your Groq API Key \n "
    " - Upload PDFs on the main page \n "
    " - Ask questions and see chat history"
)

## API Keys & embedding setup
api_key = st.sidebar.text_input("Groq API Key", type="password")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN","")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# only proceed if the user has entered their Groq Key
if not api_key:
    st.warning("🔑 Please enter your Groq API Key in the sidebar to continue.")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")

# File Uploader 
uploaded_files = st.file_uploader(
    "📁 Choose PDF file(s)",
    type="pdf",
    accept_multiple_files=True,
)

# A placeholder to collect all documents
all_docs = []

# Generate a unique session ID for the vector store if none exists
if "vector_store_id" not in st.session_state:
    st.session_state.vector_store_id = f"chroma_index_{int(time.time())}"

# Use the session-specific vector store path
vector_store_path = f"./{st.session_state.vector_store_id}"

if uploaded_files:
    # Show spinner while loading
    with st.spinner("🔁 Loading and splitting PDFs..."):
        for pdf in uploaded_files:
            # Write to a temp file so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.getvalue())
                pdf_path = tmp.name

            # Load the pdf into a list of Document objects
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            st.write(f"📄 Extracted {len(docs)} pages from {pdf.name}")
            
            all_docs.extend(docs)
            
            # Clean up the temporary file
            try:
                os.unlink(pdf_path)
            except Exception as e:
                st.warning(f"Could not remove temporary file: {e}")

    # Split docs into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(all_docs)
    st.write(f"🔍 Loaded {len(all_docs)} documents, produced {len(splits)} chunks.")
    
    if not splits:
        st.warning("No text chunks found – please upload at least one valid PDF.")
        st.stop()

    # Filter out reference pages manually - we'll prioritize content from early pages
    content_splits = []
    for doc in splits:
        page_num = int(doc.metadata.get('page', 100))
        if page_num < 30:  # Only keep pages that are likely to be main content
            # We'll add them directly to our content_splits list
            content_splits.append(doc)
    
    st.write(f"📑 Using {len(content_splits)} content chunks (excluding reference pages).")
    
    # Create a new vector store with each document upload
    # Instead of deleting the old one, create a new one with a unique name
    @st.cache_resource(show_spinner=False)
    def get_vectorstore(_splits, vs_path, version=1):
        return Chroma.from_documents(
            _splits,
            embedding=embeddings,
            persist_directory=vs_path
        )
    
    # Use our filtered content chunks instead of all splits
    vectorstore = get_vectorstore(content_splits, vector_store_path, version=1)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
    )
    
    # Build a history-aware retriever that uses past chat to refine searches
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given the chat history and the latest user question, formulate a search query 
        that will help find relevant information from the paper.
        
        Focus on retrieving content from the main body of the paper, not references or citations."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research assistant analyzing scientific papers.
        
        When answering questions:
        1. Focus on the main content of the paper (abstract, introduction, methodology, results, discussion)
        2. DO NOT use the reference section to answer questions about the paper's content
        3. If asked about authors, look for the actual paper authors, not reference authors
        4. If you don't have the relevant section in the context, say so clearly
        
        Use the retrieved context to answer:\n\n{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Session state for chat history
    if "chathistory" not in st.session_state:
        st.session_state.chathistory = {}
    
    def get_history(session_id: str):
        "Retriever of initialize chat history for a session"
        if session_id not in st.session_state.chathistory:
            st.session_state.chathistory[session_id] = ChatMessageHistory()
        return st.session_state.chathistory[session_id]
    
    # Wrap the RAG chain so it automatically logs history
    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    # Chat UI
    session_id = st.text_input("🆔 Session ID", value="default_session")
    user_question = st.chat_input("✍🏻 Your question here...")
    
    if user_question:
        try:
            history = get_history(session_id)
            result = conversational_rag.invoke(
                {"input": user_question},
                config={"configurable": {"session_id": session_id}},
            )
            answer = result["answer"]
            
            st.chat_message("user").write(user_question)
            st.chat_message("assistant").write(answer)
            
            # show full history below
            with st.expander("📖 Full chat history"):
                for msg in history.messages:
                    role = getattr(msg, "role", msg.type)  # human and assistant
                    content = msg.content
                    st.write(f"**{role.title()}: ** {content}")
            
        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")
            st.write("Please try rephrasing your question or uploading the document again.")
else:
    st.info("ℹ️ Upload one or more PDFs above to begin.")