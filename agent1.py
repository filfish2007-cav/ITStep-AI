import dotenv
import os

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    trim_messages,
)


dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

llm = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
    api_key=gemini_api_key,
)

@tool
def check_password_complexity(password: str) -> str:
    """
    Перевіряє складність паролю за встановленими критеріями.

    :param password: рядок паролю, який потрібно перевірити
    :return: текстовий звіт про те, що добре, а що погано у паролі
    """
    good = []
    bad = []

    # 1. Перевірка довжини (>8)
    if len(password) > 8:
        good.append("Довжина паролю більше 8 символів.")
    else:
        bad.append("Довжина паролю 8 або менше символів.")

    # 2. Перевірка наявності літер, цифр, спецсимволів
    has_letters = any(char.isalpha() for char in password)
    has_digits = any(char.isdigit() for char in password)
    has_special = any(not char.isalnum() for char in password)

    if has_letters:
        good.append("Пароль містить літери.")
    else:
        bad.append("Пароль не містить літер.")

    if has_digits:
        good.append("Пароль містить цифри.")
    else:
        bad.append("Пароль не містить жодної цифри.")

    if has_special:
        good.append("Пароль містить спеціальні символи.")
    else:
        bad.append("Пароль не містить спеціальних символів.")

    if has_letters:
        if any(char.islower() for char in password) and any(char.isupper() for char in password):
            good.append("Присутні літери в різних регістрах (великі та малі).")
        else:
            bad.append("Бракує літер у різних регістрах (потрібні і великі, і малі).")

    result = "Аналіз паролю:\n\n"
    if good:
        result += "ДОБРЕ:\n- " + "\n- ".join(good) + "\n\n"
    if bad:
        result += "ПОГАНО:\n- " + "\n- ".join(bad) + "\n\n"

    if not bad:
        result += "ВИСНОВОК: Пароль надійний."
    else:
        result += "ВИСНОВОК: Пароль не відповідає вимогам безпеки і потребує покращення."

    return result


## task 2
search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)


@tool
def search_person(name:str) -> str:
    """
    search persons name in internet
    :param name: str - name surname
    :return: information about person
    """

    info = str(search.results(f"last news about:{name}"))
    return info


agent = create_agent(
    model=llm,
    tools=[check_password_complexity,search_person],
)

messages = [
    SystemMessage(
        """
        You are private detective and proffesional hacker.
        Your task is either to investigate latest news about person or check hor strong is a password anything else u dont know
        """
    )
]



if __name__ == "__main__":
    while True:
        user_query = input("Ви (введіть пароль для перевірки або натисніть Enter для виходу): ")

        if user_query == '':
            break

        human_message = HumanMessage(user_query)
        messages.append(human_message)

        input_data = {
            "messages": messages
        }

        response = agent.invoke(input_data)
        messages = response['messages']

        answer = messages[-1]

        if isinstance(answer.content, list):
            text_output = "".join(block.get("text", "") for block in answer.content if isinstance(block, dict))
            print(text_output)
        else:
            print(answer.content)

        print("-" * 40)


