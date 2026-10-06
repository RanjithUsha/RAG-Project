import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

pdf_path = Path(__file__).with_name("lecture2.pdf")
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()
splitter = TokenTextSplitter(chunk_size=1500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(chunks[0].page_content)  # Print the first 500 characters of the loaded content
