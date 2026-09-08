import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

# заголовок
st.title("famous person chat bot")

# завантаження апі ключа за допомогою streamlit
api_key = st.secrets.get("GEMINI_API_KEY")

# створити llm
llm = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
    api_key=api_key,
)

# user_query = st.chat_input("Ваше повідомлення")
#
# # якщо це початок то створити історію в session state
# if user_query is None:
#     # історія повідомлень
#     st.session_state['history'] = [
#         # перше повідомлення з основними інструкціями(промпт)
#         SystemMessage(
#             """
#             Ти -- ввічливий чат бот, твоя задача давити короткі та
#             чіткі відповіді на питання
#             """
#         )
#     ]
#
# # якщо повідомлення введено, то дати відповідь від моделі
# if user_query:
#     # переволимо повідомлення в HumanMessage
#     human_message = HumanMessage(user_query)
#
#     # добавляємо до історії повідомлень
#     st.session_state['history'].append(human_message)
#
#     # запускаємо модель
#     response = llm.invoke(st.session_state['history'])
#
#     # response -- AIMessage
#     # добавляємо до історії повідомлень
#     st.session_state['history'].append(response)
#
# def get_message_text(message):
#     content = message.content
#     if isinstance(content, str):
#         return content
#     if isinstance(content, list):
#         parts = []
#         for block in content:
#             if isinstance(block, dict) and block.get("type") == "text":
#                 parts.append(block.get("text", ""))
#             elif isinstance(block, str):
#                 parts.append(block)
#         return "".join(parts)
#     return str(content)
#
# # вивести всю історію спілкування
# for message in st.session_state['history']:
#     if isinstance(message, SystemMessage):
#         continue
#
#     text = get_message_text(message)   # <-- instead of message.content
#
#     role = "human" if isinstance(message, HumanMessage) else "ai"
#
#     with st.chat_message(role):
#         st.markdown(text)


## task 1

# Завдання 1
# Напишіть додаток, який симулює спілкування з певною
# відомою людиною.
# З ким саме спілкуватись вводить користувач через
# st.text
# _input()

user_query = st.chat_input("Ваше повідомлення")
person = st.text_input("choose a role(any famous person) ")
print(person)

if 'history' not in st.session_state and person:
    # історія повідомлень
    st.session_state['history'] = [
        # перше повідомлення з основними інструкціями(промпт)
        SystemMessage(
            f"""
            Ти -- {person}, твоя задача давати відповіді на питання виключно у стилі {person}
            """
        )
    ]

print(str(SystemMessage))

if user_query:
    human_message = HumanMessage(user_query)

    st.session_state['history'].append(human_message)

    response = llm.invoke(st.session_state['history'])

    st.session_state['history'].append(response)

    for message in st.session_state['history']:
        # пропускаємо SystemMessage
        # if isinstance(message, SystemMessage):
        #     continue

        # отримати вміст
        text = message.text

        # отримати роль
        if isinstance(message, HumanMessage):
            role = "human"
        else:
            role = 'ai'

        # вивести повідомлення з підписом
        with st.chat_message(role):
            st.markdown(text)






