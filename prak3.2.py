import dotenv
import os
import langchain
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

## task 1

# завантадити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# # модель
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",  # назва моделі
    api_key=api_key  # ключ до сервера з моделлю
)

# class GenreBooks(BaseModel):
#     genre: str = Field(description="Жанр книги")
# # створюємо парсер
#
# parser = PydanticOutputParser(pydantic_object=GenreBooks)
#
# instructions = parser.get_format_instructions()
# # print(instructions)
# prompt_genre = PromptTemplate.from_template("""
# ти бібліотекар.
# твоя задача визначати жанр книг.
#
# ### ІНСТРУКЦІЯ
# Відповідь має бути до 2-3 слів
#
# ### ФОРМАТ ВІДПОВІДІ
# {format_instructions}
#
# ### ВХІДНІ ДАНІ
# {book_name}
# """,partial_variables={"format_instructions": instructions})
#
# chain = prompt_genre | llm | parser
#
# book_name = "1984"
#
# data_book = {"book": book_name}
#
# response = chain.invoke(data_book)
#
# print(response)

## task 2


# =====================================================================
# КРОК 1: Перший ланцюг — Генерація основного змісту з короткого опису
# =====================================================================

# class MainContent(BaseModel):
#     main_idea: str = Field(description="Основна мета або суть листа")
#     key_points: list[str] = Field(description="Список ключових пунктів/фактичної інформації для листа")
#
# parser1 = PydanticOutputParser(pydantic_object=MainContent)
#
# prompt1 = PromptTemplate.from_template(
#     """
#     Ти — аналітик та асистент з копірайтингу.
#     Твоя задача — прийняти короткий опис листа та виокремити з нього основний зміст і ключові тези.
#
#     ### ІНСТРУКЦІЇ ###
#     1. Не пиши готовий лист (без вітань чи підписів).
#     2. Сформулюй чітку суть повідомлення та розбий її на тези.
#
#     ### ФОРМАТ ВІДПОВІДІ ###
#     {format_instructions}
#
#     ### ВХІДНІ ДАНІ ###
#     Короткий опис: {description}
#     """,
#     partial_variables={"format_instructions": parser1.get_format_instructions()}
# )
#
# chain1 = prompt1 | llm | parser1
#
#
# # =====================================================================
# # КРОК 2: Другий ланцюг — Генерація фінального листа за змістом і стилем
# # =====================================================================
#
# class FinalLetter(BaseModel):
#     subject: str = Field(description="Тема листа")
#     letter_body: str = Field(description="Повний текст листа (включаючи привітання, основний текст та завершення)")
#
# parser2 = PydanticOutputParser(pydantic_object=FinalLetter)
#
# prompt2 = PromptTemplate.from_template(
#     """
#     Ти — професійний копірайтер та редактор.
#     Твоя задача — написати повноцінний лист, використовуючи наданий зміст та вказаний стиль.
#
#     ### ІНСТРУКЦІЇ ###
#     1. Повністю адаптуй лексику, вітання та звернення під потрібний стиль: {style}.
#     2. Використай усе наповнення з основного змісту та тез.
#     3. Якщо ім'я або назва компанії відсутні — залиш стандартні плейсхолдери на кшталт [Ім'я].
#
#     ### ФОРМАТ ВІДПОВІДІ ###
#     {format_instructions}
#
#     ### ВХІДНІ ДАНІ ###
#     Основна ідея: {main_idea}
#     Ключові тези: {key_points}
#     Бажаний стиль: {style}
#     """,
#     partial_variables={"format_instructions": parser2.get_format_instructions()}
# )
#
# chain2 = prompt2 | llm | parser2
#
#
# # =====================================================================
# # ПРИКЛАД ВИКОРИСТАННЯ
# # =====================================================================
#
# if __name__ == "__main__":
#     # Вхідні дані від користувача
#     user_description = "Попросити колегу Наталю передати звіт по проекту до п'ятниці, бо керівництво перенесло дедлайн."
#     user_style = "Неформальний, дружній, але ввічливий робочий тон"
#
#     print("--- Запуск Ланцюга 1: Аналіз опису та формування змісту ---")
#     step1_response = chain1.invoke({"description": user_description})
#
#     print(f"Основна ідея: {step1_response.main_idea}")
#     print(f"Тези: {step1_response.key_points}\n")
#
#     print("--- Запуск Ланцюга 2: Генерація листа у заданому стилі ---")
#     step2_response = chain2.invoke({
#         "main_idea": step1_response.main_idea,
#         "key_points": step1_response.key_points,
#         "style": user_style
#     })
#
#     print(f"\n[ТЕМА]: {step2_response.subject}")
#     print("=" * 50)
#     print(step2_response.letter_body)

## task 3



