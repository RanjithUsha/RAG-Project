import getpass
import os

from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStore
load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI Book"}),
    Document(page_content="Pandas is a powerful data manipulation library.", metadata={"source": "Programming Book"}),
    Document(page_content="Neural networks are a key component of deep learning.", metadata={"source": "Deep Learning Book"})
]

embeddings = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")

results = vectorstore.similarity_search("What is used in Artificial Intelligence?", k=2)

for result in results:
    print(result.page_content)
    print(result.metadata)


retriever = vectorstore.as_retriever()

docs = retriever.invoke("What is used in Artificial Intelligence?")
for d in docs:
    print(d.page_content)