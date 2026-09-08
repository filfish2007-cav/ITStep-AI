import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

# заголовок
st.title("ITStep chat bot")

# завантаження апі ключа за допомогою streamlit
api_key = st.secrets.get("GEMINI_API_KEY")

# створити llm
llm = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
    api_key=api_key,
)

user_query = st.chat_input("Ваше повідомлення")

# якщо це початок то створити історію в session state
if user_query is None:
    # історія повідомлень
    st.session_state['history'] = [
        # перше повідомлення з основними інструкціями(промпт)
        SystemMessage(
            """
            Ти -- ввічливий чат бот, твоя задача давити короткі та
            чіткі відповіді на питання
            """
        )
    ]

# якщо повідомлення введено, то дати відповідь від моделі
if user_query:
    # переволимо повідомлення в HumanMessage
    human_message = HumanMessage(user_query)

    # добавляємо до історії повідомлень
    st.session_state['history'].append(human_message)

    # запускаємо модель
    response = llm.invoke(st.session_state['history'])

    # response -- AIMessage
    # добавляємо до історії повідомлень
    st.session_state['history'].append(response)

def get_message_text(message):
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)

# вивести всю історію спілкування
for message in st.session_state['history']:
    if isinstance(message, SystemMessage):
        continue

    text = get_message_text(message)   # <-- instead of message.content

    role = "human" if isinstance(message, HumanMessage) else "ai"

    with st.chat_message(role):
        st.markdown(text)
