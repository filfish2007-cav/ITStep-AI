# Завдання 1
# Підключіть модель LLM за допомогою свого API key.
# Попросіть модель згенерувати:
# ●
# відповідь на питання у вигляді одного
# слова(наприклад яка столиця Франції?)
# ●
# код python
# ●
# коротку історію
# Підберіть параметри креативності та довжини

import os
import dotenv
import langchain
from langchain_google_genai import GoogleGenerativeAI
from sympy.physics.units import temperature

# завантаження даних з файлу .env
dotenv.load_dotenv()

# сам api key
api_key = os.getenv('GEMINI_API_KEY')

llm = GoogleGenerativeAI(
     model='gemini-3.6-flash',   # назва моделі
     api_key=api_key,
     temperature=0.1
 )


with open("/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson9/rules.txt", "r") as f:
    rules = f.read()

history = { }

while True:
    question = input("What is the question? ")
    response = llm.invoke(f""""
    ти консультант атракціону, відповідай на запитання клієнтів на основі правил {rules}
    Відповідай тільки спираючись на правила які я прикрепив 
    якщо відповіді на питання немає у правилах чи якщо немає інформації стосовно цього питання ти не знаєш відповді
    Ось запитання: {question}
    Ось попередні запитання користувача та твої відпоіді: {history}""")

    print(f"Answer: {response}")

    history[question] = response



# response = llm.invoke('What is the capital of Gabon - 1 word')
# response2 = llm.invoke('make up a story about programmer - 5 sentences')



print(response2)