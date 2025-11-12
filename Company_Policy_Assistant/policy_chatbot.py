import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document  # Fixed
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import tempfile
from typing import List

# ========================
# CONFIGURATION
# ========================
st.set_page_config(page_title="Company Policy Assistant", layout="centered")
st.title("Company Policy Assistant")

with st.sidebar:
    st.header("Settings")
    embed_model = st.text_input(
        "Embedding model", value="nomic-embed-text", help="Model name as in `ollama list`"
    )
    chat_model = st.text_input(
        "Chat model", value="llama3.2", help="Model name for answer generation"
    )
    st.markdown("---")
    st.header("Upload Policy Docs")
    uploaded_files = st.file_uploader(
        "PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

# ========================
# SESSION STATE
# ========================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ========================
# LOAD & INDEX DOCUMENTS
# ========================
@st.cache_resource(show_spinner="Indexing documents …")
def index_documents(_files, _embed_model):
    if not _files:
        return None, None

    docs: List[Document] = []
    for file in _files:
        # write to temp file
        suffix = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        # loader
        if file.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        loaded = loader.load()
        for d in loaded:
            d.metadata["source"] = file.name
        docs.extend(loaded)
        os.unlink(tmp_path)

    # chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=50, length_function=len
    )
    splits = splitter.split_documents(docs)

    # embeddings + vectorstore
    embeddings = OllamaEmbeddings(model=_embed_model)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return vectorstore, retriever


# ========================
# BUILD RAG CHAIN (Ollama)
# ========================
def build_rag_chain(retriever, _chat_model):
    llm = ChatOllama(model=_chat_model, temperature=0.0)

    template = """You are a helpful assistant for company policies.
Answer **only** using the context below. Be concise and natural.
If the answer cannot be found, reply: "I don't have information on that."

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(
            f"**Source: {d.metadata.get('source','?')}** (page {d.metadata.get('page','?')})\n{d.page_content}"
            for d in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ========================
# PROCESS UPLOADED FILES
# ========================
if uploaded_files:
    with st.spinner("Embedding & indexing …"):
        vs, ret = index_documents(uploaded_files, embed_model)
        st.session_state.vectorstore = vs
        st.session_state.retriever = ret
        st.session_state.rag_chain = build_rag_chain(ret, chat_model)
    st.success(f"Indexed {len(uploaded_files)} file(s)")

# ========================
# CHAT UI
# ========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(s)

if prompt := st.chat_input("Ask about policies …"):
    if not st.session_state.rag_chain:
        st.error("Upload at least one policy document first.")
        st.stop()

    # ---- user ----
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---- assistant ----
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            answer = st.session_state.rag_chain.invoke(prompt)

            # fetch the raw retrieved docs for citations
            raw_docs = st.session_state.retriever.invoke(prompt)
            sources = []
            for d in raw_docs:
                src = d.metadata.get("source", "unknown")
                page = d.metadata.get("page", "?")
                snippet = d.page_content[:350]
                if len(d.page_content) > 350:
                    snippet += "…"
                sources.append(f"**{src}** (page {page})\n\n{snippet}")

            st.markdown(answer)
            if sources:
                with st.expander("View Sources"):
                    for s in sources:
                        st.markdown(f"---\n{s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

# ========================
# SAMPLE POLICIES (quick test)
# ========================
with st.expander("Need sample policies?"):
    st.markdown(
        """
- **[SHRM Leave Policy (PDF)](https://www.shrm.org/resourcesandtools/tools-and-samples/policies/pages/cms_000302.aspx)**  
- **[Remote Work Policy (PDF)](https://www.flexjobs.com/blog/wp-content/uploads/2021/03/Remote-Work-Policy-Template.pdf)**  
- **[SANS IT Acceptable Use (TXT)](https://www.sans.org/media/security-awareness-training/SANS-Security-Policy-Templates.pdf)** (save as .txt)
"""
    )