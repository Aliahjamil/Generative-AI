import os
import json
import hashlib
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

BASE_DIR = "chroma_db"
REGISTRY = os.path.join(BASE_DIR, "books.json")  # collection_name -> display name
os.makedirs(BASE_DIR, exist_ok=True)

st.set_page_config(page_title="Chat with your Book", page_icon="📚", layout="wide")


# ---------- cached resources ----------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, max_tokens=1024)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ]
)


# ---------- registry helpers ----------
def load_registry():
    if os.path.exists(REGISTRY):
        with open(REGISTRY, "r", encoding="utf-8") as f:
            reg = json.load(f)
    else:
        reg = {}

    # Pick up the book you already indexed before the UI existed
    # (Chroma's default collection name is "langchain")
    if os.path.exists(os.path.join(BASE_DIR, "chroma.sqlite3")) and "langchain" not in reg:
        reg["langchain"] = "Existing book (original)"
        save_registry(reg)
    return reg


def save_registry(reg):
    with open(REGISTRY, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)


def get_store(collection_name):
    return Chroma(
        collection_name=collection_name,
        persist_directory=BASE_DIR,
        embedding_function=get_embeddings(),
    )


# ---------- indexing ----------
def index_pdf(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    collection_name = "book_" + hashlib.md5(file_bytes).hexdigest()[:16]

    registry = load_registry()
    if collection_name in registry:
        return collection_name, False  # already indexed

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=BASE_DIR,
    )

    registry[collection_name] = uploaded_file.name
    save_registry(registry)
    return collection_name, True


# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_book" not in st.session_state:
    st.session_state.active_book = None

# ---------- sidebar ----------
with st.sidebar:
    st.header("📚 Your books")

    uploaded = st.file_uploader("Upload a PDF book", type=["pdf"])
    if uploaded and st.button("Process book", use_container_width=True):
        with st.spinner("Reading, chunking and embedding... this can take a while for big books"):
            name, is_new = index_pdf(uploaded)
        st.session_state.active_book = name
        st.session_state.messages = []
        if is_new:
            st.success("Book added!")
        else:
            st.info("This book was already indexed, switched to it.")
        st.rerun()

    st.divider()

    registry = load_registry()
    if registry:
        names = list(registry.keys())
        default_idx = (
            names.index(st.session_state.active_book)
            if st.session_state.active_book in names
            else 0
        )
        selected = st.selectbox(
            "Chat with",
            options=names,
            index=default_idx,
            format_func=lambda c: registry[c],
        )

        if selected != st.session_state.active_book:
            st.session_state.active_book = selected
            st.session_state.messages = []
            st.rerun()

        top_k = st.slider("Chunks to retrieve (k)", 2, 8, 4)

        if st.button("🗑️ Delete this book", use_container_width=True):
            get_store(selected).delete_collection()
            registry.pop(selected, None)
            save_registry(registry)
            st.session_state.active_book = None
            st.session_state.messages = []
            st.rerun()

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        top_k = 4
        st.info("No books yet. Upload a PDF to get started.")

# ---------- main area ----------
st.title("📖 Chat with your Book")

registry = load_registry()
active = st.session_state.active_book

if not registry or active not in registry:
    st.write("👈 Upload a PDF in the sidebar and click **Process book**.")
    st.stop()

st.caption(f"Currently chatting with: **{registry[active]}**")

# show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**Page {s['page']}**")
                    st.caption(s["text"])

# new question
query = st.chat_input("Ask something about the book...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retriever = get_store(active).as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k, "fetch_k": max(10, top_k * 3), "lambda_mult": 0.5},
            )
            docs = retriever.invoke(query)
            context = "\n\n".join(d.page_content for d in docs)

            final_prompt = PROMPT.invoke({"context": context, "question": query})
            answer = get_llm().invoke(final_prompt).content

        st.markdown(answer)

        sources = [
            {
                "page": int(d.metadata.get("page", -1)) + 1,  # PyPDFLoader pages are 0-indexed
                "text": d.page_content[:300] + "...",
            }
            for d in docs
        ]
        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"**Page {s['page']}**")
                st.caption(s["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )