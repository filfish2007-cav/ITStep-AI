import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

# заголовок
st.title("The only language tutor chat bot u need")

# завантаження апі ключа за допомогою streamlit
api_key = st.secrets.get("GEMINI_API_KEY")

# створити llm
llm = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
    api_key=api_key,
)

user_query = st.chat_input("Ваше повідомлення")
language = st.text_input("choose a language ")
print(language)

if 'history' not in st.session_state and language:
    # історія повідомлень
    st.session_state['history'] = [
        # перше повідомлення з основними інструкціями(промпт)
        SystemMessage(
            f"""
            Ти -- мовний репетитор.

            Твоя задача -- допомагати користувачу вивчати мови.
            Користувач обрав мову пояснення: {language}.
        
            Правила:
            1. Якщо користувач просить перекласти слово або фразу:
               - дай переклад;
               - наведи приклад використання цього слова або фрази в реченні;
               - за потреби коротко поясни значення або особливості використання.
        
            2. Якщо користувач просить перекласти речення:
               - дай переклад;
               - поясни граматику речення;
               - зверни увагу на важливі граматичні конструкції, наприклад:
                 there is/there are, часи дієслів, пасивний стан, модальні дієслова,
                 умовні речення, порядок слів тощо.
        
            3. Якщо користувач ставить питання про граматику:
               - пояснюй її просто і зрозуміло;
               - наводь приклади.
        
            4. Відповідай мовою {language}, якщо користувач не попросив іншу мову.
        
            5. Не обмежуйся лише англійською мовою. Користувач може вивчати будь-яку мову.
        
            6. Якщо користувач просто спілкується з тобою мовою, яку вивчає,
               допомагай йому практикувати її та виправляй помилки з коротким поясненням.
        
            Будь ввічливим, зрозумілим та корисним мовним репетитором.
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
        if isinstance(message, SystemMessage):
            continue

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