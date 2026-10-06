"""Shared helpers for indexing PDFs and RAG Q&A."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
UPLOAD_DIR = PROJECT_ROOT / "uploads"

load_dotenv(PROJECT_ROOT / ".env")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def clean_text(text: str) -> str:
    """Remove lone UTF-16 surrogates from PDF extract text."""
    return "".join(ch for ch in text if not (0xD800 <= ord(ch) <= 0xDFFF))


def require_mistral_api_key() -> None:
    if not os.getenv("MISTRAL_API_KEY", "").strip():
        raise ValueError(
            "MISTRAL_API_KEY is missing. Add it to .env in the RAG Project folder."
        )


def get_embeddings() -> MistralAIEmbeddings:
    require_mistral_api_key()
    return MistralAIEmbeddings(model="mistral-embed")


def get_llm() -> ChatMistralAI:
    require_mistral_api_key()
    return ChatMistralAI(model_name="ministral-8b-2512")


def get_vectorstore() -> Chroma:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def chunk_pdf_documents(pdf_path: Path):
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)
        chunk.metadata["source_file"] = pdf_path.name
    return [c for c in chunks if c.page_content.strip()]


def index_pdf_file(pdf_path: Path) -> int:
    """Load, chunk, embed, and persist a PDF. Returns number of chunks added."""
    chunks = chunk_pdf_documents(pdf_path)
    if not chunks:
        return 0
    store = get_vectorstore()
    store.add_documents(chunks)
    return len(chunks)


def index_uploaded_pdf(file_name: str, file_bytes: bytes) -> int:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file_name).name
    dest = UPLOAD_DIR / safe_name
    dest.write_bytes(file_bytes)
    return index_pdf_file(dest)


def chroma_has_documents() -> bool:
    if not CHROMA_DIR.exists():
        return False
    try:
        store = get_vectorstore()
        return store._collection.count() > 0
    except Exception:
        return False


def session_messages_to_langchain(messages: list[dict]) -> list[BaseMessage]:
    history: list[BaseMessage] = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer using the provided document context "
            "and the conversation so far. If the context does not contain the answer, "
            "say you don't know.",
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "Context from uploaded documents:\n{context}\n\nQuestion: {question}",
        ),
    ]
)


def answer_with_rag(
    question: str,
    chat_history: list[dict],
    *,
    k: int = 3,
) -> tuple[str, list[str]]:
    store = get_vectorstore()
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "lambda_mult": 0.5, "fetch_k": max(k * 3, 10)},
    )
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = list(
        dict.fromkeys(
            doc.metadata.get("source_file") or doc.metadata.get("source", "unknown")
            for doc in docs
        )
    )

    llm = get_llm()
    prompt_value = RAG_PROMPT.invoke(
        {
            "chat_history": session_messages_to_langchain(chat_history),
            "context": context or "(No matching document chunks found.)",
            "question": question,
        }
    )
    response = llm.invoke(prompt_value)
    return response.content, sources