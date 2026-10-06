from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai  import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI

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

retriever = vectorstore.as_retriever()

llm = ChatMistralAI(model_name="ministral-8b-2512")


multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

query = "What is gradient descent?"
docs = multiquery_retriever.invoke(query)

print("\nMulti-Query Search Results:\n")

for doc in docs:
    print(doc.page_content)