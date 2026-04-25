from deepeval.test_case import LLMTestCase
# Импортируем нашу кастомную метрику (предполагается, что она сохранена в code_coherence.py)
from code_text_coherence import CodeCoherenceMetric 
from giga_chat import CustomGigaChat

# 1. We prepare the test data (output data from the LLM being tested)
# Here we specifically make the first part coherent and the second part meaningless, 
# so that yandex.metrica gives you a score below 1.0 and explains the reason.
mock_llm_output = """
# Инструкция по запуску проекта

Для начала вам необходимо запустить локальный сервер. Используйте следующий скрипт:

```python
import http.server
import socketserver

PORT = 8080
Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("Сервер запущен на порту", PORT)
    httpd.serve_forever()
Этот код импортирует встроенные модули Python и поднимает простой HTTP-сервер на порту 8080. Он будет раздавать файлы из той директории, откуда был запущен скрипт.

Далее создадим главную страницу:

<!DOCTYPE html>
<html>
<body>
    <h1>Добро пожаловать</h1>
</body>
</html>
Для того чтобы подключиться к базе данных PostgreSQL, вам нужно использовать библиотеку psycopg2. Обязательно проверьте настройки вашего firewall, чтобы порт 5432 был открыт.
"""

# 2. Initialize GigaChat
gigachat_llm = CustomGigaChat(credentials="api")

# 3. Creating a metric with a custom LLM as a judge
#threshold=0.5 means that the test will be passed (success=True) if at least 50% of the code is described correctly.
metric = CodeCoherenceMetric(
    threshold=0.5,
    model=gigachat_llm,
    include_reason=True,
    verbose_mode=True
)


# 4. Creating a test case
test_case = LLMTestCase(
    input="Напиши инструкцию по запуску локального сервера и созданию стартовой HTML страницы.",
    actual_output=mock_llm_output
)

#5. Measuring Coherence
print("Запуск оценки Code Coherence...")
metric.measure(test_case)

#6. Print results
print(f"Успешно: {metric.is_successful()}")
print(f"Итоговая оценка: {metric.score}")
print(f"Причина: {metric.reason}")