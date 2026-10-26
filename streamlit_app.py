import os
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ----------------------------- Configuration -----------------------------
PERSIST_DIRECTORY = "chroma_db"
MODEL_NAME = "ministral-8b-2512"

st.set_page_config(
    page_title="DocuMind | RAG Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Styling -----------------------------------
st.markdown(
    """
    <style>
      .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #10182b 55%, #111827 100%);
        color: #eef2ff;
      }
      [data-testid="stSidebar"] {
        background: #0a0f1c;
        border-right: 1px solid rgba(148, 163, 184, .18);
      }
      .hero {
        padding: 1.5rem 1.7rem;
        border: 1px solid rgba(148, 163, 184, .22);
        border-radius: 22px;
        background: linear-gradient(120deg, rgba(30, 41, 59, .94), rgba(30, 41, 59, .45));
        margin-bottom: 1.25rem;
      }
      .hero h1 { margin: 0; font-size: 2.15rem; color: #f8fafc; }
      .hero p { margin: .55rem 0 0; color: #cbd5e1; font-size: 1rem; }
      .pill {
        display: inline-block; padding: .28rem .65rem; margin-bottom: .65rem;
        border-radius: 999px; background: rgba(99, 102, 241, .18);
        color: #c7d2fe; border: 1px solid rgba(129, 140, 248, .35);
        font-size: .78rem; font-weight: 600;
      }
      div[data-testid="stChatMessage"] {
        border: 1px solid rgba(148, 163, 184, .15);
        border-radius: 16px; padding: .5rem .7rem;
        background: rgba(15, 23, 42, .55);
      }
      .small-muted { color: #94a3b8; font-size: .88rem; }
      div.stButton > button {
        border-radius: 10px; font-weight: 600; min-height: 2.7rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- RAG setup ---------------------------------
@st.cache_resource
def get_embedding_model():
    return MistralAIEmbeddings()


@st.cache_resource
def get_llm():
    return ChatMistralAI(model=MODEL_NAME)


def get_vectorstore():
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embedding_model(),
    )


def get_retriever():
    return get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
Use only the provided context to answer the question.
If the answer is not present in the context, say:
"I could not find the answer in the context provided."
Do not invent facts. When useful, mention the page number of the supporting information.""",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}",
        ),
    ]
)


def index_pdf(uploaded_file):
    """Load, split, and index an uploaded PDF into the persistent Chroma store."""
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if file_hash in st.session_state.indexed_files:
        return 0, "This PDF has already been indexed in this session."

    # PyPDFLoader works with a file path, so write the upload to a temporary file.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        documents = PyPDFLoader(temp_path).load()
    finally:
        Path(temp_path).unlink(missing_ok=True)

    if not documents:
        raise ValueError("No readable pages were found in this PDF.")

    # Preserve the uploaded filename in metadata and retain PDF page numbers.
    for document in documents:
        document.metadata["source"] = uploaded_file.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text could be extracted. The PDF may be scanned or image-only.")

    get_vectorstore().add_documents(chunks)
    st.session_state.indexed_files.add(file_hash)
    return len(chunks), None


def answer_question(question):
    docs = get_retriever().invoke(question)
    if not docs:
        return "I could not find the answer in the context provided.", []

    context_parts = []
    for doc in docs:
        page = doc.metadata.get("page")
        page_label = f"Page {page + 1}" if isinstance(page, int) else "Page unknown"
        source = doc.metadata.get("source", "Uploaded document")
        context_parts.append(f"[Source: {source}, {page_label}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)
    final_prompt = PROMPT.invoke({"context": context, "question": question})
    response = get_llm().invoke(final_prompt)
    return response.content, docs


# ----------------------------- Session state -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

# ----------------------------- Sidebar -----------------------------------
with st.sidebar:
    st.markdown("## DocuMind")
    st.caption("Your documents. Your questions. Context-grounded answers.")
    st.divider()

    st.markdown("### Add a PDF")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text-based PDFs work best. Scanned PDFs need OCR before their text can be searched.",
    )

    if st.button("Process PDFs", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one PDF first.")
        elif not os.getenv("MISTRAL_API_KEY"):
            st.error("MISTRAL_API_KEY is missing. Add it to your .env file and restart Streamlit.")
        else:
            progress = st.progress(0, text="Preparing documents…")
            total = len(uploaded_files)
            success_count = 0
            for index, pdf in enumerate(uploaded_files):
                try:
                    chunk_count, message = index_pdf(pdf)
                    if message:
                        st.info(f"**{pdf.name}:** {message}")
                    else:
                        st.success(f"**{pdf.name}** indexed into {chunk_count} chunks.")
                        success_count += 1
                        if pdf.name not in st.session_state.uploaded_names:
                            st.session_state.uploaded_names.append(pdf.name)
                except Exception as exc:
                    st.error(f"Could not process **{pdf.name}**: {exc}")
                progress.progress((index + 1) / total, text=f"Processed {index + 1} of {total}")
            if success_count:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Ready! I indexed {success_count} PDF document(s). Ask me a question about their contents.",
                    }
                )

    st.divider()
    st.markdown("### Knowledge base")
    if st.session_state.uploaded_names:
        st.caption("Indexed during this session")
        for name in st.session_state.uploaded_names:
            st.markdown(f"• {name}")
    else:
        st.markdown('<p class="small-muted">No PDFs indexed in this session yet.</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### Current setup")
    st.markdown(
        "- **Chat model:** Mistral\n"
        "- **Embeddings:** Mistral\n"
        "- **Vector store:** ChromaDB\n"
        "- **Retrieval:** MMR (4 chunks)"
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------- Main view ---------------------------------
st.markdown(
    """
    <div class="hero">
      <span class="pill">RETRIEVAL-AUGMENTED GENERATION</span>
      <h1>Ask your documents.</h1>
      <p>Upload PDFs, retrieve relevant passages, and get answers grounded in your own knowledge base.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric("Documents this session", len(st.session_state.uploaded_names))
with metric_col2:
    st.metric("Retrieval strategy", "MMR")
with metric_col3:
    st.metric("Answer model", "Mistral AI")

if not os.getenv("MISTRAL_API_KEY"):
    st.warning("Add `MISTRAL_API_KEY=your_key` to your `.env` file before processing PDFs or asking questions.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your indexed PDFs…")
if question:
    if not os.getenv("MISTRAL_API_KEY"):
        st.error("MISTRAL_API_KEY is missing. Add it to your `.env` file and restart the app.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents and composing an answer…"):
                try:
                    answer, source_docs = answer_question(question)
                    st.markdown(answer)
                    if source_docs:
                        with st.expander("Sources used"):
                            for doc in source_docs:
                                page = doc.metadata.get("page")
                                page_label = f"Page {page + 1}" if isinstance(page, int) else "Page unknown"
                                st.markdown(f"**{doc.metadata.get('source', 'Uploaded document')} · {page_label}**")
                                excerpt = doc.page_content.strip()
                                st.caption(excerpt[:700] + ("…" if len(excerpt) > 700 else ""))
                except Exception as exc:
                    answer = f"Something went wrong while generating the answer: {exc}"
                    st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
