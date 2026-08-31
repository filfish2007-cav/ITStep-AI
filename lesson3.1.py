import dotenv
import os

import langchain
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# завантадити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# # модель
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",  # назва моделі
    api_key=api_key  # ключ до сервера з моделлю
)

# Користувач задає питання.
# Потрібно дати відповіть та запропопонувати цікаві факти по тій
# же темі що і питання


# # варіант 1 -- все в один промпт

prompt = PromptTemplate.from_template("""
    Ти -- вчитель.
    Твоя задача давати короткі відповіді на питання та пропонувати
    цікаві факти на схожу тему

    ###ІНСТРУКЦІЇ###
    1. Відповідь має бути до 3 речень
    2. Запропонуй до 5 цікавих фактів на ту ж тему що і питання
    3. Факти мають зацікавити учня дізнатись більше

    ###ВХІДНІ ДАНІ###
    Питання: {question}

    Відповідь:

""")

user_question = "Коли був політ на місяць?"


# # вставляємо питання в промпт
# data = {
#     "question": user_question
# }
#
# text = prompt.invoke(data)
# # print(text)
#
# # виклик моделі
# response = llm.invoke(text)
# print(response)
#
# # через те що в нас ChatGoogleGenerativeAI варто виводити відповідь ось так
# text_response = response.content[0]["text"]
# print(text_response)


# об'єднуємо в один крок
# створення ланцюга

# chain = prompt | llm
#
# data = {
#     "question": user_question
# }
# response = chain.invoke(data)
#
# text_response = response.content[0]["text"]
#
# print(text_response)


# варіант 2 -- розьити на 2 кроки
# дати відповідь та визначити тему питання
# згенерувати цікаві факти по темі


# структура відповіді
class AnswerTheme(BaseModel):
    answer: str = Field(description="відповідь на питання")
    themes: list[str] = Field(description="список тем пов'язаних з питанням користувача")


# створення парсер
parser = PydanticOutputParser(pydantic_object=AnswerTheme)

# інструкція від парсера
instrustions = parser.get_format_instructions()

# промпт
prompt1 = PromptTemplate.from_template("""
    Ти -- чатбот по навчанню.
    Твоя задача дати відповідь на питання користувача і також визначити список
    тем пов'язаних з питання

    ###ІНСТРУНКЦІЇ###
    1. Відповідь має бути до 3 речень
    2. Має бути не більше 5 тем

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instrustions}

    ###ВХІДНІ ДАНІ###
    Питання: {question}
""",
                                       partial_variables={"format_instrustions": instrustions}
                                       # оодразу передає інструкції від парсера
                                       )

# ланцюг для першого кроку
chain1 = prompt1 | llm | parser

# використання
user_question = "Коли був політ на місяць?"

data = {
    "question": user_question,
}


# response = chain1.invoke(data)
#
# answer = response.answer
# print(answer)
#
# themes = response.themes
# print(type(themes))
# print(themes)


# крок 2
# згенерувати цікаві факти по темі

class Facts(BaseModel):
    facts: list[str] = Field(description="список цікавих вактів на задані теми")


parser = PydanticOutputParser(pydantic_object=Facts)

instrustions = parser.get_format_instructions()

# промпт

prompt = PromptTemplate.from_template("""
    Ти -- викладач.
    Твоя задача навести декілька цікавих фактів на задані теми що зацікавити
    студента.

    ###ІНСТРУКЦІЇ###
    1. сам факт має бути одним речень
    2. Не більше 3 фактів

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ВХІДНІ ДАНІ###
    Список тем: {themes}
""",
                                      partial_variables={"format_instructions": instrustions}
                                      )

# ланцюг для кроку 2

chain2 = prompt | llm | parser

# використання

user_question = "Коли був політ на місяць?"

# дані для першого ланцюга
data = {
    "question": user_question
}

# запускаємо перший ланцюг

response1 = chain1.invoke(data)

print(f"Відповідь на питання: {response1.answer}")

# # дані для другого ланцюга
data = {
    "themes": response1.themes
}

# запускаємо другий ланцюг
response2 = chain2.invoke(data)

print(f"Цікаві Факти:")
for fact in response2.facts:
    print(fact)


