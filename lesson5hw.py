import os
import dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")


llm = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
    api_key=gemini_api_key,
)

places_searcher = GoogleSerperAPIWrapper(
    serper_api_key=serper_api_key,
    type="places"
)


def recommend_restaurants(query: str) -> str:
    """
    Шукає інформацію про ресторани за запитом користувача.

    :param query: запит для пошуку (наприклад, "італійські ресторани в Києві")
    :return: список ресторанів з їхньою назвою, посиланням на сайт та рейтингом
    """
    print(f"[Debug] Виклик інструменту пошуку ресторанів із запитом: {query}")

    # Отримуємо результати у вигляді словника
    results = places_searcher.results(query)

    places = results.get('places', [])

    if not places:
        return "На жаль, за вашим запитом ресторанів не знайдено."

    formatted_response = []

    for place in places[:5]:
        name = place.get('title', 'Назва не вказана')
        rating = place.get('rating', 'Рейтинг відсутній')
        website = place.get('website', 'Сайт відсутній')

        formatted_response.append(
            f"Назва: {name}\nРейтинг: {rating}\nСайт: {website}\n"
        )

    return "\n".join(formatted_response)


agent = create_agent(
    model=llm,
    tools=[recommend_restaurants]
)

messages = [
    SystemMessage(
        """
        Ти ввічлий чат-бот, який допомагає користувачам знаходити найкращі ресторани. 
        Твоя задача давати інформативні та чіткі рекомендації.

        У тебе є доступ до інструменту:
        * recommend_restaurants -- використовуй його завжди, коли користувач просить порадити ресторан, кафе чи бар.

        Завжди красиво оформлюй відповідь, вказуючи назву, рейтинг та посилання.
        """
    )
]

print("Бот запущений. Напишіть свій запит або натисніть Enter для виходу.")

while True:
    user_query = input("\nВи: ")

    if user_query == '':
        print("Робота завершена.")
        break

    human_message = HumanMessage(user_query)
    messages.append(human_message)

    input_data = {
        "messages": messages
    }

    try:
        response = agent.invoke(input_data)

        messages = response['messages']

        answer = messages[-1]
        print("\nБот:")
        print(answer.content)

    except Exception as e:
        print(f"\nСталася помилка: {e}")