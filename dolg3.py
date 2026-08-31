import dotenv
import os
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    trim_messages,
)

dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)

# ==========================================
# Завдання 1: Чат-бот для рольового спілкування
# ==========================================

print("--- Завдання 1: Рольовий чат-бот ---")
messages1: list[BaseMessage] = [
    SystemMessage("""
    Ти -- акторський чатбот. Твоя задача -- спілкуватися з користувачем у стилі персонажа книги, фільму або відомої людини, якого він вкаже.

    ###ІНСТРУКЦІЇ###
    1. Визнач персонажа з першого повідомлення користувача і підтримуй його стиль.
    2. Якщо персонаж або книга невідомі, дай відповідь, що ця інформація тобі невідома, та запропонуй 3-4 відомих приклади на вибір (наприклад: Дарт Вейдер, Шерлок Холмс, Альберт Ейнштейн).
    """)
]

while True:
    user_text = input("Ви: ")
    if not user_text: break
    messages1.append(HumanMessage(content=user_text))
    response1 = llm.invoke(messages1)
    print(f"AI: {response1.content}")
    messages1.append(response1)


# ==========================================
# Завдання 2: Чат-бот з умов повернення товару (з історією)
# ==========================================

print("--- Завдання 2: Бот підтримки (повернення товару) ---")
messages2: list[BaseMessage] = [
    SystemMessage("""
    Ти -- агент служби підтримки інтернет-магазину.

    ###ІНСТРУКЦІЇ###
    1. Відповідай ТІЛЬКИ на питання стосовно умов та процедури повернення товару.
    2. Якщо користувач запитує щось інше, відповідай чітко: "На жаль, у мене немає інформації з цього питання. Я можу допомогти лише з умовами повернення товару."
    """)
]

trimmer2 = trim_messages(
    strategy='last',
    token_counter=len,
    max_tokens=5,
    start_on='human',
    end_on='human',
    include_system=True
)

chain2 = trimmer2 | llm

while True:
    user_text = input("Ви: ")
    if not user_text: break
    messages2.append(HumanMessage(content=user_text))
    response2 = chain2.invoke(messages2)
    print(f"AI: {response2.content}")
    messages2.append(response2)


# ==========================================
# Завдання 3: Чат-бот для вивчення англійської
# ==========================================

print("--- Завдання 3: Вчитель англійської ---")
messages3: list[BaseMessage] = [
    SystemMessage("""
    Ти -- професійний вчитель англійської мови.

    ###ІНСТРУКЦІЇ###
    1. Якщо користувач просить перекласти слово або фразу: надай переклад та приклад використання в реченні англійською.
    2. Якщо користувач просить перекласти ціле речення: надай переклад речення та пояснення граматичних структур (наприклад: there is/are, часові форми тощо).
    """),
    HumanMessage(content="Переклади слово 'book'"),
    AIMessage(content="Переклад: книга. Приклад використання: I am reading a fascinating book right now."),
    HumanMessage(content="Переклади речення 'There are many apples on the table'"),
    AIMessage(
        content="Переклад: На столі багато яблук. Граматика: Конструкція 'there are' використовується для вказівки на наявність або місцезнаходження предметів у множині (many apples).")
]

while True:
    user_text = input("Ви: ")
    if not user_text: break
    messages3.append(HumanMessage(content=user_text))
    response3 = llm.invoke(messages3)
    print(f"AI: {response3.content}")
    messages3.append(response3)


# ==========================================
# Завдання 4: Вчитель англійської зі збереженням слів у JSON
# ==========================================

print("--- Завдання 4: Вчитель англійської з пам'яттю ---")

learned_words = []
json_file_path = "learned_words.json"

if os.path.exists(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        learned_words = json.load(f)

extractor_messages: list[BaseMessage] = [
    SystemMessage("""
    Твоя задача -- знайти всі унікальні англійські слова в наданому тексті та повернути їх у вигляді списку через кому, без жодних додаткових символів чи пояснень. 
    Тільки англійські слова в початковій формі.
    """)
]

system_prompt4 = f"""
Ти -- розумний вчитель англійської мови. 
Список вже вивчених користувачем слів: {', '.join(learned_words) if learned_words else 'Користувач ще не вивчив жодного слова'}.

###ІНСТРУКЦІЇ###
1. Якщо користувач просить перекласти слово або фразу: надай переклад та приклад використання в реченні, але використовуй ТІЛЬКИ ті англійські слова, які є у списку вивчених слів (плюс саме нове слово).
2. Якщо користувач просить перекласти ціле речення: надай переклад і додатково поясни значення всіх невідомих слів з цього речення (тих, яких немає у списку вивчених).
"""

messages4: list[BaseMessage] = [SystemMessage(system_prompt4)]

trimmer4 = trim_messages(
    strategy='last',
    token_counter=len,
    max_tokens=5,
    start_on='human',
    end_on='human',
    include_system=True
)

chain4 = trimmer4 | llm

while True:
    user_text = input("Ви: ")
    if not user_text: break

    messages4.append(HumanMessage(content=user_text))
    response4 = chain4.invoke(messages4)
    print(f"AI: {response4.content}")
    messages4.append(response4)

    extract_msgs = extractor_messages + [HumanMessage(content=response4.content)]
    words_response = llm.invoke(extract_msgs)
    new_words = [w.strip().lower() for w in words_response.content.split(',')]

    for word in new_words:
        if word and word not in learned_words:
            learned_words.append(word)

    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(learned_words, f, ensure_ascii=False, indent=4)

    messages4[0] = SystemMessage(f"""
    Ти -- розумний вчитель англійської мови. 
    Список вже вивчених користувачем слів: {', '.join(learned_words)}.

    ###ІНСТРУКЦІЇ###
    1. Якщо користувач просить перекласти слово або фразу: надай переклад та приклад використання в реченні (складай речення ТІЛЬКИ з вивчених слів та цільового слова).
    2. Якщо користувач просить перекласти ціле речення: надай переклад і додатково поясни значення всіх невідомих слів з цього речення.
    """)