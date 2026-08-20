import os
import dotenv

from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Завантаження API-ключа
dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Ініціалізація LLM
llm = GoogleGenerativeAI(
    model='gemini-1.5-flash',
    api_key=api_key,
    temperature=0
)

# Читання файлу політики повернення
file_path = os.path.join("data", "lesson9", "return_policy.txt")
try:
    with open(file_path, "r", encoding="utf-8") as f:
        policy_text = f.read()
except FileNotFoundError:
    # Резервний текст для демонстрації працездатності коду
    policy_text = """
    1. Повернення товару можливе протягом 14 днів з моменту покупки.
    2. Необхідно зберегти чек та оригінальне пакування.
    3. Товар не повинен мати слідів використання.
    4. Гроші повертаються протягом 3-5 робочих днів на банківську картку покупця.
    """

# Формування інструкції за структурою з image_b5d820.png
instruction = f"""[РОЛЬ] Ти є оператором служби підтримки.
[КОНТЕКСТ] Ситуація така: клієнти звертаються із запитаннями щодо умов повернення товару.
[ЗАВДАННЯ] Твоя задача — консультувати клієнтів.
[ВХІДНІ ДАНІ] Ось дані (документ з правилами): {policy_text}
[ОБМЕЖЕННЯ] Правила: давай відповіді виключно на основі вхідних даних. Якщо відповіді в тексті немає, повідом про це.
[ФОРМАТ] Відповідай у форматі стислого тексту без зайвих вітань."""

# Створення шаблону промпту з підтримкою історії
prompt = PromptTemplate.from_template("""Instruction: {instruction}
{history}
Human: {user_input}
AI:""")

chain = prompt | llm

# Змінна для зберігання історії діалогу
chat_history = ""

print("Бот готовий. Введіть питання або залиште рядок порожнім для виходу.\n")

# Основний цикл чату
while True:
    user_input = input("Human: ")

    # Завершення діалогу при порожньому введенні
    if not user_input.strip():
        break

    # Генерація відповіді
    response = chain.invoke({
        "instruction": instruction,
        "history": chat_history,
        "user_input": user_input
    })

    ai_response = response.strip()
    print(f"AI: {ai_response}\n")

    # Оновлення історії у заданому форматі
    if chat_history:
        chat_history += "\n"
    chat_history += f"Human: {user_input}\nAI: {ai_response}"