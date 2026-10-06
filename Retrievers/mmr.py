from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in ML", metadata={"source": "doc1"}),
    Document(page_content="Gradient descent minimizes the loss function.", metadata={"source": "doc2"}),
    Document(page_content="Gradient descent is an optimization that minimizes the Loss function.", metadata={"source": "doc4"}),    
    Document(page_content="Neural Netorks use gradient descent for training.", metadata={"source": "doc4"}),
    Document(page_content="Support Vector Machines are supervised learning algorithms.", metadata={"source": "doc4"}),
]

embeddings = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(docs, embeddings)

similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}) 

print("\nSimilarity Search Results:\n")

similarity_docs = similarity_retriever.invoke("What is gradient descent?")

for doc in similarity_docs:
    print(doc.page_content)


mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

print("\nMMR Search Results:\n")

mmr_docs = mmr_retriever.invoke("What is gradient descent?")

for doc in mmr_docs:
    print(doc.page_content)