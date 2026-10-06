import sys

from langchain_community.retrievers import ArxivRetriever

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

retriever = ArxivRetriever(
    load_max_docs=5,
    load_all_available_meta=True,
)

docs = retriever.invoke("Large Language Models")

for i, doc in enumerate(docs, start=1):
    print(f"\nResult {i}:")
    print(f"Title: {doc.metadata.get('Title')}")
    print(f"Authors: {doc.metadata.get('Authors')}")
    print(f"Published: {doc.metadata.get('Published')}")
    print(f"Summary: {doc.page_content}")
