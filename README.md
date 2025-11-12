# Company Policy Assistant: Local RAG Chatbot with Ollama
A **Retrieval-Augmented Generation (RAG)** chatbot designed to answer questions about company policies using uploaded **PDF or TXT documents**. This project demonstrates the use of **LangChain**, **Ollama LLM & embeddings**, and **Chroma vector database** to provide **contextual answers with source citations**.

# 🏆 Features

✅ Upload **PDF** or **TXT** files containing company policies.

✅ Semantic search with embeddings + vector database (Chroma).

✅ LLM-based answer generation using **Ollama**.

✅ Retrieves relevant document chunks and cites sources.

✅ Maintains conversation history within the session.

🌐 Interactive web UI built with Streamlit.

# 🎯 Objectives

This project is designed to showcase skills in:

1. Document ingestion and chunking.
2. Embedding & semantic search for relevant content.
3. Retrieval-Augmented Generation (RAG).
4. Session-based conversation memory.
5. Interactive, user-friendly UI for querying documents.

# 🛠 Tech Stack
| Component            | Tool / Library                          |
| -------------------- | --------------------------------------- |
| **LLM & Embeddings** | [Ollama](https://ollama.com/)           |
| **Vector Store**     | [Chroma](https://www.trychroma.com/)    |
| **Python Framework** | [LangChain](https://www.langchain.com/) |
| **Web Interface**    | [Streamlit](https://streamlit.io/)      |
| **Document Loaders** | `PyPDFLoader` (PDF), `TextLoader` (TXT) |

# ⚙️ Installation & Libraries

## 1. Install Ollama
[Download here](https://ollama.com/download)

### 2. Pull Models
```python
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2
```

## 3. Clone the Repository
```python
git clone https://github.com/MirazulHasan/COMPANY_POLICY_ASSISTANT_CHATBOT.git
cd company-policy-chatbot
```
## 4. Create a Virtual Environment
```python
python -m venv venv
```
### Activate the environment

### Linux / macOS:
```python
source venv/bin/activate
```
### Windows:
```python
venv\Scripts\activate
```
## 5. Install Required Python Libraries

You can install all dependencies using `pip`:
```python
pip install -r requirements.txt
```
Or manually:
```python
pip install streamlit
pip install langchain
pip install langchain_community
pip install langchain_text_splitters
pip install langchain_chroma
pip install langchain_ollama
pip install chromadb
pip install pypdf
```
> Python 3.11+ recommended.

## 6. Run Ollama Locally
```python
ollama serve
```
> Required for embeddings and chat.

## 7. Start the Streamlit App
```python
streamlit run policy_chatbot.py
```
Open your browser: http://localhost:8501

# 📝 Usage

### 1. Upload Policy Documents

- Click **"Upload Policy Docs"** in the sidebar.

- Accepts **PDF** or **TXT** files.

### 2. Ask Questions

Use the chat input to ask anything like:

> What is the company’s leave policy?

> How can employees request remote work?


### 3. View Sources

- Answers come **only from uploaded docs.**

- Click **"View Sources"** to see file + page + snippet.

### 4. Conversation History

- Maintained within the session.

- Each message shows in the chat interface.

# 🧩 Sample Policy Documents (for testing)

- [Local Initiatives Support Corporation](https://www.lisc.org/media/filer_public/93/dd/93ddc43d-3917-4361-8381-70c1f0a5de54/sample_policy_and_procedures_manual.pdf)

- [Template.net](https://images.template.net/wp-content/uploads/2022/07/Policy-and-Procedure-PDF.pdf)

- [The People in Dairy](https://thepeopleindairy.org.au/wp-content/uploads/2019/02/workplace_policies_procedures_v1.pdf)

- [MD Accountants & Auditors Inc.](https://www.mdacc.co.za/wp-content/uploads/2016/04/04-Policy-Manual-Apr-2016-excl-SAICA.pdf)

# 💡 How it Works

1. **Load Documents →** Read PDFs/TXTs
2. **Split →** Chunk into 1000-char segments with overlap
3. **Embed →** Convert to vectors using `nomic-embed-text`
4. **Store →** In Chroma vector DB
5. **Query →** Retrieve top-4 relevant chunks
6. **Generate →** LLM answers using only retrieved context
7. **Cite →** Show source file + page + snippet
8. **Chat →** Session memory preserved

# 📁 Files
- `policy_chatbot.py` – Main Streamlit + RAG app
- `requirements.txt` – Python dependencies

# 📌 Notes
- Ensure `nomic-embed-text` and `llama3.2` are pulled via Ollama.
- Session memory only (in-browser). For persistence: add SQLite or JSON logging.
- Fully **local & private** — no data leaves your machine.

# 📚 References

- [LangChain Documentation](https://www.langchain.com/docs/)

- [Chroma Vector Database](https://www.trychroma.com/)

- [Ollama](https://ollama.com/)

- [Streamlit Docs](https://docs.streamlit.io/)

# 💻 License

**© 2025 Md. Mirazul Hasan**

**All Rights Reserved.**

*For educational and internal use.*
