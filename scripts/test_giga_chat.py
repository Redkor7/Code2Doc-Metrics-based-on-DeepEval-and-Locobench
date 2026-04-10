from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from giga_chat import CustomGigaChat

# Инициализируем GigaChat
gigachat_llm = CustomGigaChat(credentials="api_k")

# Создаём метрику с кастомной LLM
metric = AnswerRelevancyMetric(model=gigachat_llm)

# Пример тест-кейса
test_case = LLMTestCase(
    input="Какая погода в Москве?",
    actual_output="Сегодня в Москве солнечно, +15°C."
)

# Измеряем
metric.measure(test_case)
print(metric.score)
print(metric.reason)