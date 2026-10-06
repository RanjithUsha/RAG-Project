import sys
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# chunk_size caps merged pieces; "\n\n" never appears in notes.txt.txt, so CharacterTextSplitter kept the whole file as one chunk
splitter = RecursiveCharacterTextSplitter(separators=[""], chunk_size=10, chunk_overlap=1)

notes = Path(__file__).with_name("notes.txt.txt")
loader = TextLoader(notes, encoding="utf-8")
docs = loader.load()
chunks = splitter.split_documents(docs)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"--- chunk {i} ({len(chunk.page_content)} chars) ---")
    print(chunk.page_content)
    print()
    print()
    print()
