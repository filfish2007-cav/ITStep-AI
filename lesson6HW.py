# створення агентів
# агент -- чат-бот(llm) + інструменти

import dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_agent
from langchain_core.tools import tool
from pinecone import Pinecone
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)

embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-001",
    api_key=api_key,
)

pc = Pinecone(api_key=pinecone_api_key)

index_name = "itset-docs"

index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding
)


@tool
def document_search(query: str):
    """
    Пошук документів у векторній базі даних.

    База даних містить інформацію з файлу
    huge_file.txt про умови користування Google.

    :param query: запит від користувача
    :return: схожі документи
    """

    results = vector_store.similarity_search(
        query,
        k=2,
    )

    return results


# створення агента
agent = create_agent(
    model=llm,
    tools=[document_search],
)


# системний промпт
messages = [
    SystemMessage("""
    Ти -- ввічливий чат-бот.

    ### ІНСТРУКЦІЯ ###

    1. Ти відповідаєш на питання користувача про умови
       користування сервісами Google.

    2. Для питань, пов'язаних з умовами користування Google,
       ОБОВ'ЯЗКОВО використовуй document_search.

    3. Відповідай на основі інформації, знайденої в базі даних.

    4. Не вигадуй інформацію, якої немає у знайдених документах.

    5. Якщо потрібної інформації немає в базі даних,
       повідом про це користувачу.
    """)
]


# цикл спілкування
while True:

    # Запит від користувача
    user_query = input("Ви: ")

    # умова закінчення
    if user_query == "":
        break

    # зробити human message
    user_message = HumanMessage(user_query)

    # додаємо повідомлення в історію
    messages.append(user_message)

    # отримати відповідь від агента
    data = {
        "messages": messages
    }

    data = agent.invoke(data)

    # отримуємо нову історію повідомлень
    messages = data["messages"]

    # відповідь моделі -- останнє повідомлення
    response = messages[-1]

    # вивести відповідь
    print(response.text)

    # виведення історії
    print()
    print("----------ІСТОРІЯ-----------")

    for message in messages:
        print(repr(message))

    print("-----------------------------")
    print()