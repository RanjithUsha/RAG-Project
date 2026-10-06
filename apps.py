"""Streamlit RAG app: one PDF per vector DB, chat scoped to that document only."""

from __future__ import annotations

import streamlit as st

from rag_utils import (
    CHROMA_DIR,
    answer_with_rag,
    chroma_has_documents,
    create_vector_database_from_upload,
    require_mistral_api_key,
)

st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="wide")

st.title("📚 RAG Document Chat")
st.caption(
    "Upload a PDF, create a vector database for that file only, then ask questions "
    "about that document."
)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_pdf" not in st.session_state:
        st.session_state.active_pdf = None


init_session_state()

with st.sidebar:
    st.header("Vector database")
    st.write(f"Store path: `{CHROMA_DIR}`")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None:
        if st.button("Create vector database", type="primary", use_container_width=True):
            try:
                require_mistral_api_key()
                with st.spinner("Building vector database for this PDF…"):
                    count, pdf_name = create_vector_database_from_upload(
                        uploaded.name, uploaded.getvalue()
                    )
                if count == 0:
                    st.warning("No text could be extracted from this PDF.")
                    st.session_state.active_pdf = None
                else:
                    st.session_state.active_pdf = pdf_name
                    st.session_state.messages = []
                    st.success(
                        f"Vector database ready: **{count}** chunks from `{pdf_name}`."
                    )
                    st.info("Chat history was cleared for the new document.")
            except Exception as exc:
                st.error(str(exc))

    active = st.session_state.active_pdf
    if active and chroma_has_documents():
        st.info(f"Active document: **{active}**")
        st.caption("Questions use only this PDF—not other files you uploaded before.")
    else:
        st.warning("Upload a PDF and click **Create vector database** to start.")

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

try:
    require_mistral_api_key()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption(f"Document: {', '.join(message['sources'])}")

active_pdf = st.session_state.active_pdf
chat_placeholder = (
    f"Ask a question about {active_pdf}…"
    if active_pdf
    else "Upload a PDF and create a vector database first…"
)
question = st.chat_input(chat_placeholder)

if question:
    if not active_pdf or not chroma_has_documents():
        st.warning("Create a vector database from your PDF before asking questions.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching this document and generating answer…"):
            try:
                prior = st.session_state.messages[:-1]
                answer, sources = answer_with_rag(
                    question,
                    prior,
                    document_name=active_pdf,
                )
            except Exception as exc:
                answer = f"Something went wrong: {exc}"
                sources = []

        st.markdown(answer)
        if sources:
            st.caption(f"Document: {', '.join(sources)}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
