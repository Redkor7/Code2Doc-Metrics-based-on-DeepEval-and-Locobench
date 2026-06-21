from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from giga_chat import CustomGigaChat
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GigaChat_API_Key")

gigachat_llm = CustomGigaChat(credentials=API_KEY)

metric = AnswerRelevancyMetric(model=gigachat_llm)

test_case = LLMTestCase(
    input="Какая погода в Москве?",
    actual_output="Сегодня в Москве солнечно, +15°C."
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)