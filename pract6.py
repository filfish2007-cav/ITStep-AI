import os
import dotenv
from typing import List

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages, BaseMessage
)
from uuid import uuid4


dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash',
    api_key=gemini_api_key,
)


embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=gemini_api_key,
)

# створення весторної бази даних
pc = Pinecone(api_key=pinecone_api_key)
index_name = "soup"  # назва бази даних

if pc.has_index(index_name):
    pc.delete_index(index_name)

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,      # кількість чисел при кодування
        metric="cosine",    # формула для схожості
        spec=ServerlessSpec(
            cloud="aws",         # хмарний сервер(амазон)
            region="us-east-1"   # регіон(Каліфорнія)
        ),
    )

index = pc.Index(index_name)
vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding
)

with open('/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson_rag/files/future_of_ai.txt', "r", encoding="utf-8") as f:
    text1 = f.read()

doc1 = Document(
    page_content=text1,   # вміст дукумента
    metadata={               # додаткова інформація
        "type": "future",
        "author": "Anton Halysh"
    }
)

with open('/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson_rag/files/intro.txt', "r", encoding="utf-8") as f:
    text2 = f.read()

doc2 = Document(
    page_content=text2,   # вміст дукумента
    metadata={               # додаткова інформація
        "type": "intro",
        "author": "Anton Halysh"
    }
)

with open('/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson_rag/files/machine_learning.txt', "r", encoding="utf-8") as f:
    text3 = f.read()

doc3 = Document(
    page_content=text3,  # вміст дукумента
    metadata={  # додаткова інформація
        "type": "machine_learning",
        "author": "Anton Halysh"
    }
)

with open('/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson_rag/files/neural_networks.txt', "r", encoding="utf-8") as f:
    text4 = f.read()

doc4 = Document(
    page_content=text4,  # вміст дукумента
    metadata={  # додаткова інформація
        "type": "neural_networks",
        "author": "Anton Halysh"
    }
)

docs = [doc1, doc2, doc3, doc4]

ids = [str(uuid4()) for _ in range(len(docs))]

vector_store.add_documents(
     documents=docs,
     ids=ids
 )

@tool
def search_doc(user_query: str) -> List[Document]:
    """
    Шукає схожі документи з релевантної інформацією до запиту користувача

    :param user_query: запит користувача
    :return: список документів з релевантною інформацією
    """
    result_docs = vector_store.similarity_search(
        user_query,  # текст для порівняння схожості
        k=1,         # кількість документів у відповіді
    )

    return result_docs


# створення агента
agent = create_agent(
    model=llm,  # мовна модель
    tools=[search_doc]
)

# історія повідомлень + інструкції

messages = [
    SystemMessage(
        """
        Ти ввічлий чат-бот. Твоя задача давати інформативні та чіткі відповіді
        на запити користувача.

        У тебе є доступ до таких інструментів:
        * search_doc -- цукає інформацію в базі даних що містить:
            * інформація про майбутнє ШІ
            * інформація про визначення ШІ
            * інформація про навчання машин
            * інформація про нейроні мережі
        """
    )
]

while True:
    user_query = input("Ви: ")

    if user_query == '':
        break

    # переводимо str рядок у  HumanMessage
    human_message = HumanMessage(user_query)

    # добавляємо повідослення користувача до історії
    messages.append(human_message)

    # застосування агента
    # треба передавати словник
    input_data = {
        "messages": messages
    }

    response = agent.invoke(input_data)
    # response -- словник з усією історією + відповідь моделі

    # отримання всіє історії повідомлень
    messages = response['messages']

    # отримати фінальну відповідь моделі
    answer = messages[-1]
    print(answer.content)

    # виведемння всієї історії
    print()
    print("Історія")

    for message in messages:
        print(repr(message))



