import dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    trim_messages,
)

# Завантаження змінних оточення
dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Ініціалізація моделі
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",  # Використовуйте актуальну назву моделі
    api_key=api_key
)

# Системне повідомлення з інструкціями для рольового відіграшу
system_prompt = """
Ти — професійний актор-чатбот. Твоя мета — імітувати стиль спілкування різних персонажів (книг, фільмів) або відомих людей.

### ІНСТРУКЦІЇ ###
1. Визнач з повідомлення користувача, ким саме ти маєш бути.
2. Якщо персонаж тобі відомий: повністю перейми його манеру мовлення, характерні фрази, тон та світогляд. Відповідай виключно від його імені.
3. Якщо персонаж невідомий, вигаданий користувачем або не вказаний:
   - Повідом, що не маєш інформації про цього персонажа.
   - Запропонуй на вибір 4-5 відомих варіантів (наприклад: Майстер Йода, Шерлок Холмс, Тоні Старк, Альберт Ейнштейн, Тарас Шевченко).
4. Залишайся в ролі, поки користувач явно не попросить змінити персонажа.
5. Відповіді мають бути лаконічними, якщо інше не вимагається стилем самого персонажа.
"""

messages = [
    SystemMessage(content=system_prompt)
]

# Налаштування трімера для збереження контексту (останні 5 повідомлень)
trimmer = trim_messages(
    strategy='last',
    token_counter=len,
    max_tokens=5,
    start_on='human',
    end_on='human',
    include_system=True
)

# Створення ланцюга обробки
chain = trimmer | llm

print("Чат-бот запущено. Напишіть, з ким хочете поспілкуватися (наприклад: 'Будь як Шерлок Холмс. Що думаєш про сучасні технології?').\nВведіть порожній рядок для виходу.\n")

# Цикл спілкування
while True:
    user_text = input("Ви: ")

    if user_text.strip() == "":
        break

    # Додавання повідомлення користувача
    human_message = HumanMessage(content=user_text)
    messages.append(human_message)

    # Отримання відповіді
    response = chain.invoke(messages)

    # Виведення результату
    if isinstance(response.content, list):
        text = response.content[0].get('text', '')
    else:
        text = response.content

    print(f"AI: {text}")

    # Збереження відповіді в історію
    messages.append(response)