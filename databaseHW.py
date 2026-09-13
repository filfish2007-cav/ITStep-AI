from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

import os
import dotenv
from uuid import uuid4
import json



dotenv.load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")



embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-001",
    api_key=gemini_api_key
)



pc = Pinecone(api_key=pinecone_api_key)

index_name = "itset-docs"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

file_name = "huge_file.txt"
file_path = "data/lesson_rag/huge_file.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()


# ---------------------------------------------------------
# розділення файлу на блоки
#
# між блоками є два порожніх рядки
# ---------------------------------------------------------

blocks = text.split("\n\n\n")

blocks = [
    block.strip()
    for block in blocks
    if block.strip()
]


print("Кількість блоків:", len(blocks))

docs = []

for block in blocks:

    lines = block.splitlines()

    block_name = lines[0].strip()

    doc = Document(
        page_content=block,
        metadata={
            "file_name": file_name,
            "block_name": block_name
        }
    )

    docs.append(doc)

ids = [
    str(uuid4())
    for _ in range(len(docs))
]


print("Кількість створених ID:", len(ids))

vector_store.add_documents(
    documents=docs,
    ids=ids
)


print("Документи додані в Pinecone!")


json_data = []

for doc, doc_id in zip(docs, ids):

    json_data.append({
        "id": doc_id,
        "file_name": doc.metadata["file_name"],
        "block_name": doc.metadata["block_name"]
    })


json_path = "data/lesson_rag/huge_file_ids.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        json_data,
        file,
        ensure_ascii=False,
        indent=4
    )


print("ID збережені у:", json_path)
