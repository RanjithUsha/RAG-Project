from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embeddings = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.5, "fetch_k": 10}
)

llm = ChatMistralAI(model_name="ministral-8b-2512")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that provides concise and accurate answers "
            "based on the provided context. If the answer is not present in the "
            "context, say: I don't know.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}\nAnswer:"),
    ]
)

print("RAG system created")

print(" press 0 to exit")   

while True:
    query = input("Enter your question: ")
    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])


    final_prompt = prompt.invoke({"context": context, "question": query})

    response = llm.invoke(final_prompt)

    print(f"\n AI Response: {response.content}\n")
