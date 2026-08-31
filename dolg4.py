import dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)


def summarize_conversation(chat_history: list[BaseMessage]) -> str:
    conversation_text = ""
    for msg in chat_history:
        role = "Користувач" if isinstance(msg, HumanMessage) else "AI"
        conversation_text += f"{role}: {msg.content}\n"

    summary_messages = [
        SystemMessage("""
        Ти -- помічник, який підсумовує історію чату.
        Твоя задача підсумувати всю надану розмову в декілька речень. 
        Збережи якомога більше деталей, фактів та ключових моментів з розмови.
        """),
        HumanMessage(content=f"Ось розмова:\n{conversation_text}")
    ]

    summary_response = llm.invoke(summary_messages)
    return summary_response.content


messages: list[BaseMessage] = [
    SystemMessage("""
    Ти -- ввічливий чатбот.
    Твоя задача підтримувати спілкування з користувачем.
    Якщо в історії є підсумок попередньої розмови, враховуй його.

    ###ІНСТРУКЦІЇ###
    1. Відповіді мають бути короткими та змістовними.
    """)
]

while True:
    user_text = input("Ви: ")

    if user_text == "":
        break

    human_message = HumanMessage(content=user_text)

    if len(messages[1:]) > 4:
        summary = summarize_conversation(messages[1:])

        messages = [
            messages[0],
            AIMessage(content=f"[ПІДСУМОК ПОПЕРЕДНЬОГО СПІЛКУВАННЯ]: {summary}")
        ]

    messages.append(human_message)

    response = llm.invoke(messages)

    print(f"AI: {response.content}")

    messages.append(response)

    print()
    print("----------------------------------")
    print(f"HISTORY (Кількість повідомлень: {len(messages)})")
    for message in messages:
        print(message)
    print("----------------------------------")
    print()