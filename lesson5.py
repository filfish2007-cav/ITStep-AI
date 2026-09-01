# створення агентів
# агент -- чат-бот(llm) + інструменти

import os
import dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages, BaseMessage
)


# завантаження апі ключа
dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

# створити llm
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    api_key=gemini_api_key,
    serper_api_key=serper_api_key,
)

# інструмент -- функція
# обов'язкова документація

def product(a: int, b: int) -> int:
    """
    Множить 2 цілих числа то повертає їхній добуток

    :param a: перше число
    :param b: друге число
    :return: добуток чисел
    """
    print("hello from product")
    return a * b


def get_weather(city: str, time: str) -> str:
    """
    Повертає інформацію про погоду у місті в певний час доби

    :param city: назва міста
    :param time: час доби(наприклад ранок, вечір, 10:30, 4 години дня)
    :return: інформація про погоду
    """
    print("hello from get_weather")
    return f"У {city} о {time} буде сонячно"


# інструмент для пошуку в інтернеті
searcher = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)

def search(query: str) -> str:
    """
    Шукає інформацію в інтернеті за запитом користувача

    :param query: запит користувача
    :return: результати пошуку
    """

    results = searcher.results(query)
    print(results)  # результати пошуку

    return str(results)


# створення агента
agent = create_agent(
    model=llm,   # мовна модель
    tools=[product, get_weather, search]
)

# історія повідомлень + інструкції

messages = [
    SystemMessage(
        """
        Ти ввічлий чат-бот. Твоя задача давати інформативні та чіткі відповіді
        на запити користувача.
        
        У тебе є доступ до таких інструментів:
        * product
        * get_weather
        * search -- завжди давай посилання на новини
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
    answear = messages[-1]
    print(answear.content)

    # виведемння всієї історії
    print()
    print("Історія")

    for message in messages:
        print(repr(message))




