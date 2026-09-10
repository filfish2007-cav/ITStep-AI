import os
import json
import uuid
import dotenv
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
    BaseMessage
)

# Завантаження апі ключів
dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    api_key=gemini_api_key,
)

embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    api_key=gemini_api_key,
)

pc = Pinecone(api_key=pinecone_api_key)
index_name = "soup"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

index = pc.Index(index_name)
vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding
)

# ---------------------------------------------------------
# ЗАВДАННЯ 1: Створення бази даних та JSON з ID
# ---------------------------------------------------------
files_dir = "data/lesson_rag/files"
json_mapping_file = "document_ids.json"


def initialize_database():
    if not os.path.exists(files_dir):
        print(f"Директорія {files_dir} не знайдена. Створіть її та додайте файли.")
        return

    if os.path.exists(json_mapping_file):
        print("База даних вже ініціалізована (знайдено файл document_ids.json).")
        return

    documents = []
    doc_ids = []
    file_id_mapping = {}

    for filename in os.listdir(files_dir):
        file_path = os.path.join(files_dir, filename)

        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            doc = Document(page_content=content, metadata={"source": file_path})

            doc_id = str(uuid.uuid4())

            documents.append(doc)
            doc_ids.append(doc_id)

            file_id_mapping[filename] = doc_id

    if documents:
        vector_store.add_documents(documents=documents, ids=doc_ids)

        with open(json_mapping_file, "w", encoding="utf-8") as f:
            json.dump(file_id_mapping, f, ensure_ascii=False, indent=4)
        print("База успішно ініціалізована. Дані завантажено.")


# ---------------------------------------------------------
# ЗАВДАННЯ 3: Оновлення змінених файлів в БД
# ---------------------------------------------------------
def update_modified_files(modified_filenames: List[str]):
    if not os.path.exists(json_mapping_file):
        print("Неможливо оновити: JSON файл відсутній.")
        return

    with open(json_mapping_file, "r", encoding="utf-8") as f:
        file_id_mapping = json.load(f)

    ids_to_delete = []
    new_documents = []
    new_doc_ids = []

    for filename in modified_filenames:
        if filename in file_id_mapping:
            old_id = file_id_mapping[filename]
            ids_to_delete.append(old_id)

            file_path = os.path.join(files_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                new_doc = Document(page_content=content, metadata={"source": file_path})
                new_id = str(uuid.uuid4())

                new_documents.append(new_doc)
                new_doc_ids.append(new_id)

                # Оновлюємо словник
                file_id_mapping[filename] = new_id

    if ids_to_delete:
        vector_store.delete(ids=ids_to_delete)
        print(f"Видалено старі документи з ID: {ids_to_delete}")

    if new_documents:
        vector_store.add_documents(documents=new_documents, ids=new_doc_ids)

        with open(json_mapping_file, "w", encoding="utf-8") as f:
            json.dump(file_id_mapping, f, ensure_ascii=False, indent=4)
        print(f"Додано оновлені файли: {modified_filenames}")


# Запуск ініціалізації та оновлення
initialize_database()

changed_files_list = []
update_modified_files(changed_files_list)


# ---------------------------------------------------------
# ЗАВДАННЯ 2: Інструмент та Агент
# ---------------------------------------------------------

def search_doc(user_query: str) -> List[Document]:
    """
    Шукає схожі документи з релевантною інформацією до запиту користувача.

    :param user_query: запит користувача
    :return: список документів з релевантною інформацією
    """
    result_docs = vector_store.similarity_search(
        user_query,
        k=2,
    )
    return result_docs


agent = create_react_agent(
    model=llm,
    tools=[search_doc]
)

messages = [
    SystemMessage(
        """
        Ти ввічлий чат-бот. Твоя задача давати інформативні та чіткі відповіді
        на запити користувача.

        У тебе є доступ до таких інструментів:
        * search_doc -- шукає інформацію в базі даних що містить:
            * інформація про суп
            * інформація про здоров'я

        При відповіді спирайся на отриману з бази даних інформацію.
        """
    )
]

print("\n--- Чат-бот запущено. Напишіть свій запит. ---")
while True:
    user_query = input("\nВи: ")

    if user_query.strip() == '':
        break

    human_message = HumanMessage(content=user_query)
    messages.append(human_message)

    input_data = {
        "messages": messages
    }

    response = agent.invoke(input_data)

    messages = response['messages']
    answer = messages[-1]

    print("\nБот:", answer.content)

    print("\n--- Історія повідомлень ---")
    for message in messages[-3:]:
        print(repr(message))