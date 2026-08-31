import dotenv
import os

import langchain
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)


# ==========================================
# Завдання 1: Модель для рекомендації книг
# ==========================================

# Крок 1: Отримати назву книги та визначити жанр
class BookGenre(BaseModel):
    genre: str = Field(description="жанр заданої книги")


parser1_1 = PydanticOutputParser(pydantic_object=BookGenre)

prompt1_1 = PromptTemplate.from_template("""
    Ти -- літературний експерт.
    Твоя задача визначити жанр книги за її назвою.

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ВХІДНІ ДАНІ###
    Назва книги: {book_title}
""",
                                         partial_variables={"format_instructions": parser1_1.get_format_instructions()}
                                         )
chain1_1 = prompt1_1 | llm | parser1_1


# Крок 2: Отримати назву книги, жанр та повернути список схожих книг
class SimilarBooks(BaseModel):
    books: list[str] = Field(description="список схожих книг того ж самого жанру та інших")


parser1_2 = PydanticOutputParser(pydantic_object=SimilarBooks)

prompt1_2 = PromptTemplate.from_template("""
    Ти -- бібліотекар.
    Твоя задача порекомендувати список схожих книг на основі назви книги та її жанру.

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ВХІДНІ ДАНІ###
    Назва книги: {book_title}
    Жанр: {genre}
""",
                                         partial_variables={"format_instructions": parser1_2.get_format_instructions()}
                                         )
chain1_2 = prompt1_2 | llm | parser1_2


# Використання
# data1_1 = {"book_title": "Гаррі Поттер і філософський камінь"}
# response1_1 = chain1_1.invoke(data1_1)
# data1_2 = {"book_title": "Гаррі Поттер і філософський камінь", "genre": response1_1.genre}
# response1_2 = chain1_2.invoke(data1_2)


# ==========================================
# Завдання 2: Модель для генерації листа
# ==========================================

# Крок 1: Отримати короткий опис листа та згенерувати основний зміст
class LetterContent(BaseModel):
    main_content: str = Field(description="основний зміст листа згенерований на основі опису")


parser2_1 = PydanticOutputParser(pydantic_object=LetterContent)

prompt2_1 = PromptTemplate.from_template("""
    Ти -- професійний копірайтер.
    Твоя задача згенерувати основний зміст листа на основі його короткого опису.

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ВХІДНІ ДАНІ###
    Опис листа: {letter_description}
""",
                                         partial_variables={"format_instructions": parser2_1.get_format_instructions()}
                                         )
chain2_1 = prompt2_1 | llm | parser2_1


# Крок 2: Отримати основний зміст та стиль листа та згенерувати фінальний лист
class FinalLetter(BaseModel):
    letter: str = Field(description="повністю згенерований лист у відповідному стилі")


parser2_2 = PydanticOutputParser(pydantic_object=FinalLetter)

prompt2_2 = PromptTemplate.from_template("""
    Ти -- майстер листування.
    Твоя задача написати фінальний лист, використовуючи наданий основний зміст та дотримуючись заданого стилю.

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ВХІДНІ ДАНІ###
    Основний зміст: {main_content}
    Стиль листа: {style}
""",
                                         partial_variables={"format_instructions": parser2_2.get_format_instructions()}
                                         )
chain2_2 = prompt2_2 | llm | parser2_2


# Використання
# data2_1 = {"letter_description": "Запрошення на співбесіду на посаду Python розробника"}
# response2_1 = chain2_1.invoke(data2_1)
# data2_2 = {"main_content": response2_1.main_content, "style": "формальний"}
# response2_2 = chain2_2.invoke(data2_2)


# ==========================================
# Завдання 3: Модель для генерації резюме
# ==========================================

# Крок 1: Отримати опис вакансії та повернути основні необхідні навички
class RequiredSkills(BaseModel):
    skills: list[str] = Field(description="список основних навичок, які необхідні для вакансії")


parser3_1 = PydanticOutputParser(pydantic_object=RequiredSkills)

prompt3_1 = PromptTemplate.from_template("""
    Ти -- HR спеціаліст.
    Твоя задача проаналізувати опис вакансії та виділити список основних навичок, які вимагаються від кандидата.

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ВХІДНІ ДАНІ###
    Опис вакансії: {job_description}
""",
                                         partial_variables={"format_instructions": parser3_1.get_format_instructions()}
                                         )
chain3_1 = prompt3_1 | llm | parser3_1


# Крок 2: Отримати основні навички та опис кандидата і згенерувати резюме
class GeneratedResume(BaseModel):
    resume: str = Field(description="готове згенероване резюме, адаптоване під навички вакансії")


parser3_2 = PydanticOutputParser(pydantic_object=GeneratedResume)

prompt3_2 = PromptTemplate.from_template("""
    Ти -- експерт з написання резюме.
    Твоя задача згенерувати професійне резюме для кандидата, акцентуючи увагу на необхідних навичках.

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ВХІДНІ ДАНІ###
    Необхідні навички: {skills}
    Опис кандидата: {candidate_description}
""",
                                         partial_variables={"format_instructions": parser3_2.get_format_instructions()}
                                         )
chain3_2 = prompt3_2 | llm | parser3_2

# Використання
# data3_1 = {"job_description": "Шукаємо Data Scientist з досвідом роботи в Python, Pandas та LangChain"}
# response3_1 = chain3_1.invoke(data3_1)
# data3_2 = {"skills": response3_1.skills, "candidate_description": "Маю 3 роки досвіду в аналізі даних, працював з LLM та Python"}
# response3_2 = chain3_2.invoke(data3_2)