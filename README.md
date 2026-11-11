# YouTube Multilingual RAG with LangChain & ChromaDB

A pure **LangChain** implementation for Q&A over YouTube videos using **BAAI/bge-m3** embeddings, **ChromaDB**, and **Google Gemini** (`gemini-3.5-flash-lite`), with **FastAPI** and **Streamlit**.

No custom wrapper classes, no dataclasses, and no complex manual runtime setups.

---

## Architecture (LangChain LCEL Pipeline)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load YouTube transcripts directly into LangChain Documents
docs = load_youtube_transcript_documents("https://youtu.be/...")

# 2. Split with RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# 3. Embed with BAAI/bge-m3 and store in Chroma
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
db = Chroma.from_documents(chunks, embeddings, persist_directory="./data/chroma")

# 4. Construct LangChain LCEL RAG chain
retriever = db.as_retriever(search_kwargs={"k": 5})
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.2)
prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:
{context}

Question: {question}
""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is explained in this video?")
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key

```bash
# Linux / macOS
export GOOGLE_API_KEY="your-gemini-api-key"

# Windows PowerShell
$env:GOOGLE_API_KEY = "your-gemini-api-key"
```

### 3. Run the Streamlit UI

```bash
streamlit run app.py
```

### 4. Run the FastAPI Backend (optional)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger docs: `http://localhost:8000/docs`

---

## Project Structure

```
├── app.py                  # Streamlit UI (LangChain LCEL powered)
├── api.py                  # FastAPI REST API
├── config.py               # Central constants (model names, chunk size)
├── requirements.txt        # Standard pip packages
├── test_retrieval.py       # Terminal test script for LangChain pipeline
│
├── ingestion/
│   ├── transcripts.py      # YouTube transcript loader returning LangChain Documents
│   └── chunking.py         # RecursiveCharacterTextSplitter chunking
│
├── retrieval/
│   ├── embedder.py         # HuggingFaceEmbeddings for BAAI/bge-m3
│   └── store.py            # LangChain Chroma vector store & retriever
│
├── generation/
│   └── answer.py           # Pure LCEL chain (Retriever | Prompt | LLM | Parser)
│
└── tests/
    └── test_units.py       # Fast unit test suite
```

---

## FastAPI Endpoints

| Method   | Endpoint               | Description                                                            |
| -------- | ---------------------- | ---------------------------------------------------------------------- |
| `POST`   | `/ingest`              | Load, split with `RecursiveCharacterTextSplitter`, and store in Chroma |
| `POST`   | `/chat`                | Run LCEL RAG chain and return answer + timestamp citations             |
| `GET`    | `/captions/{video_id}` | Inspect available caption tracks                                       |
| `DELETE` | `/videos/{video_id}`   | Delete video chunks from Chroma                                        |
| `GET`    | `/health`              | Health check endpoint                                                  |
