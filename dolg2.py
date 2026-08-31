import dotenv
import os

import langchain
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# модель
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)

# Крок 1: Отримати мету тренування та повернути список вправ
class ExerciseList(BaseModel):
    exercises: list[str] = Field(description="список вправ, які відповідають заданій меті тренування")


parser1 = PydanticOutputParser(pydantic_object=ExerciseList)

prompt1 = PromptTemplate.from_template("""
    Ти -- професійний фітнес-тренер.
    Твоя задача проаналізувати мету тренування користувача та скласти список відповідних вправ.

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ВХІДНІ ДАНІ###
    Мета тренування: {goal}
""",
                                       partial_variables={"format_instructions": parser1.get_format_instructions()}
                                       )
chain1 = prompt1 | llm | parser1


# Крок 2: Отримати список вправ, рівень підготовки, час на тиждень і повернути план тренувань
class TrainingPlan(BaseModel):
    plan: str = Field(description="детальний персональний план тренувань")


parser2 = PydanticOutputParser(pydantic_object=TrainingPlan)

prompt2 = PromptTemplate.from_template("""
    Ти -- професійний фітнес-тренер та експерт з планування.
    Твоя задача скласти детальний план тренувань, враховуючи заданий список вправ, рівень підготовки користувача та доступний час на тиждень.

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ВХІДНІ ДАНІ###
    Список вправ: {exercises}
    Рівень підготовки: {level}
    Час на тиждень (в годинах): {hours_per_week}
""",
                                       partial_variables={"format_instructions": parser2.get_format_instructions()}
                                       )
chain2 = prompt2 | llm | parser2

# Використання
# data1 = {"goal": "набір м'язової маси"}
# response1 = chain1.invoke(data1)

# data2 = {
#     "exercises": response1.exercises,
#     "level": "середній",
#     "hours_per_week": "4"
# }
# response2 = chain2.invoke(data2)