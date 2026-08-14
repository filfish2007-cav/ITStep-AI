# LLM
# Large Language Model
# велика мовна модель

# завантеження api key як змінну середовища
import os
import dotenv

# завантаження даних з файлу .env
dotenv.load_dotenv()

# сам api key
api_key = os.getenv('GEMINI_API_KEY')

# сама модель LMM
import langchain
from langchain_google_genai import GoogleGenerativeAI

#print(langchain.__version__)
# # root/
# #   - langchain.py
# #   - langchain_google_genai.py
#
# # # створення моделі
llm = GoogleGenerativeAI(
     model='gemini-3.5-flash-lite',   # назва моделі
     api_key=api_key
 )
# #
# # # запуск моделі
response = llm.invoke('Привіт, що таке recursive CTEs in short?')
# #
print(response)
#
#
# # Як це працює
#
# # Запит: Привіт, що таке LLM?
# # Шматок Відповіді: Привіт! LLM
#
# # Завдання моделі -- згенерувати наступне слово
# # Для кожного відомого слова генеруються ймовірності
# # розшивровується  - 30%
# # це               - 25%
# # використовується - 10%
# # яблуко           - 0.0000000001%
#
#
# # параметри креативності
# llm = GoogleGenerativeAI(
#     model='gemini-2.5-flash-lite',   # назва моделі
#     api_key=api_key,
#     top_k=10,   # вибрати випадково наступне слово з 10 з найбільшою ймовірністю
#     top_p=0.8,  # залишити ті слова, сума ймовірностей яких не менше 80%, та вибирати серед них
#     temperature=1.5  # вища температура -- відсотки стають більш однаковими
# )
#
# # temperature
# # 0 - 0.3    -- низька креативність(відповіді як по методичці)
# # 0.7 - 1.2  -- середня креативність(відповідає як людина)
# # 1.5-1.7    -- висока креативність(вигадає щось цікаве або збреше)
# # >2         -- випадкові слова