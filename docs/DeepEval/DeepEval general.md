## Метрики DeepEval — краткий обзор

### Сквозные и компонентные метрики

Поддерживают как **end-to-end** оценку всего LLM-приложения, так и **покомпонентную** проверку (RAG, агенты, отдельные вызовы). Работают с любой LLM, статистическими методами или локальными NLP-моделями.
###  Кастомные метрики

#### G-Eval

LLM-as-a-judge с Chain-of-Thoughts. Самый гибкий вариант — описываете критерии на естественном языке, LLM генерирует шаги оценки и выставляет score. Поддерживает нормализацию через logprobs и рубрики.

#### DAG (Deep Acyclic Graph)

Детерминированное дерево решений. Строите ациклический граф из узлов (TaskNode, BinaryJudgementNode, NonBinaryJudgementNode, VerdictNode), граф выполняется, выбирая путь на основе LLM-суждений. Максимальный контроль над логикой оценки.
### RAG-метрики

| Метрика                                              | Что оценивает                                                                                         |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Релевантность ответа (Answer Relevancy)              | Насколько ответ соответствует вопросу (без учёта контекста)                                           |
| Достоверность (Faithfulness)                         | Подкреплён ли каждый факт в ответе предоставленным контекстом                                         |
| Контекстуальное воспроизведение (Contextual Recall]) | Насколько полно ответ использует информацию из контекста                                              |
| Контекстуальная точность (Contextual Precision)      | Все ли факты в контексте релевантны вопросу                                                           |
| Контекстуальная релевантность (Contextual Relevancy) | Насколько контекст соответствует вопросу                                                              |
| RAGAS                                                | Композитная метрика, объединяющая достоверность, релевантность ответа и контекстуальную релевантность |
### Агентские метрики

| Метрика                   | Что оценивает                                                         |
| ------------------------- | --------------------------------------------------------------------- |
| Завершенность задачи      | Успешно ли агент выполнил поставленную задачу                         |
| Корректность инструментов | Правильно ли агент выбрал и использовал доступные инструменты/функции |
### Базовые метрики

| Метрика      | Что оценивает                                          |
| ------------ | ------------------------------------------------------ |
| Галлюцинации | Выдумывает ли LLM факты, отсутствующие в контексте     |
| Суммаризация | Качество сжатия текста (информативность, лаконичность) |
| Смещение     | Наличие нежелательных предубеждений в ответах          |
| Токсичность  | Присутствие оскорбительного или вредного контента      |
### Диалоговые метрики

| Метрика               | Что оценивает                                                            |
| --------------------- | ------------------------------------------------------------------------ |
| Сохранение знаний     | Удерживает ли модель информацию из предыдущих сообщений                  |
| Полнота диалога       | Охвачены ли все необходимые аспекты разговора                            |
| Релевантность диалога | Ответы соответствуют контексту беседы                                    |
| Соблюдение роли       | Придерживается ли модель заданной роли (например, "врач", "консультант") |
### Дополнительные возможности

#### Создание собственных метрик

Автоматическая интеграция с экосистемой DeepEval. Можно наследовать `BaseMetric` или использовать G-Eval/DAG.

#### Генерация синтетических данных

Автоматическое создание наборов данных для тестирования.

####  Безопасность (red-teaming)

40+ уязвимостей, проверяемых в несколько строк кода:
- Токсичность
- Смещение
- SQL-инъекции
- Prompt-инъекции
- и другие

10+ стратегий атак, включая продвинутые техники внедрения вредоносных промптов.

#### CI/CD интеграция

Бесшовная вставка в любой пайплайн (GitHub Actions, Jenkins, etc.).

#### Сравнение LLM на бенчмарках

Менее 10 строк кода для тестирования любой LLM на популярных бенчмарках:
- **MMLU** — мультизадачное понимание
- **HellaSwag** — здравый смысл
- **DROP** — арифметические рассуждения
- **BIG-Bench Hard** — сложные задачи
- **TruthfulQA** — правдивость ответов
- **HumanEval** — генерация кода
- **GSM8K** — математические рассуждения

---

### Итог: когда что использовать

|Сценарий|Рекомендуемая метрика|
|---|---|
|Оценка RAG-системы|Контекстуальная релевантность + Достоверность|
|Субъективная оценка качества|G-Eval с кастомными критериями|
|Чёткие правила оценки|DAG (детерминированное дерево)|
|Агенты с инструментами|Завершенность задачи + Корректность инструментов|
|Чат-боты|Диалоговые метрики (сохранение знаний, соблюдение роли)|
|Безопасность|Red-teaming (уязвимости и атаки)|
|Выбор модели|Сравнение на бенчмарках|
## Введение в LLM-оценку (LLM Evals)

### Что такое оценка LLM?

**Оценка (Evaluation)** — процесс тестирования выходных данных LLM-приложения. Включает три основных компонента:

1. **Тест-кейсы (Test Cases)** — примеры взаимодействия с LLM
2. **Метрики (Metrics)** — критерии оценки
3. **Датасеты (Evaluation Dataset)** — коллекция тестовых примеров
### Два типа оценки в DeepEval

|Тип|Описание|
|---|---|
|**End-to-end**|Оценка всей системы целиком (вход → выход)|
|**Component-level**|Оценка отдельных компонентов внутри системы|

Оба типа выполняются через:
- `deepeval test run` — в CI/CD пайплайнах
- `evaluate()` — в Python-скриптах
#### Запуск тестов
`deepeval test run test_example.py`
### Компоненты DeepEval

#### 1. Метрики (Metrics)

30+ готовых метрик, большинство использует LLM-as-a-judge:

```python
from deepeval.metrics import AnswerRelevancyMetric
answer_relevancy_metric = AnswerRelevancyMetric()
```
#### 2. Тест-кейсы (Test Cases)

Представляют одно LLM-взаимодействие:
```python
from deepeval.test_case import LLMTestCase
test_case = LLMTestCase(
    input="Who is the current president of the USA?",
    actual_output="Joe Biden",
    retrieval_context=["Joe Biden serves as the current president of America."]
)
# Запуск метрики
answer_relevancy_metric.measure(test_case)
print(answer_relevancy_metric.score)  # 0.0 - 1.0
```

**Основные поля:**
- `input` — пользовательский запрос
- `actual_output` — ответ LLM
- `expected_output` — ожидаемый ответ
- `retrieval_context` — контекст для RAG
- `tools_called` — использованные инструменты
#### 3. Датасеты (Datasets)

Коллекция `Golden` (эталонных примеров) для централизованной оценки:

```python
from deepeval.dataset import EvaluationDataset, Golden
from deepeval import evaluate
# Создание датасета
dataset = EvaluationDataset(goldens=[
    Golden(input="Question 1"),
    Golden(input="Question 2"),
])
# Преобразование goldens в test cases
for golden in dataset.goldens:
    dataset.add_test_case(
        LLMTestCase(
            input=golden.input,
            actual_output=your_llm_app(golden.input)
        )
    )
# Оценка всего датасета
evaluate(test_cases=dataset.test_cases, metrics=[answer_relevancy_metric])

```
#### 4. Синтезатор (Synthesizer)

Генерация синтетических датасетов из документов:
```python
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
synthesizer = Synthesizer()
# Генерация золотых примеров из документов
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=['example.txt', 'example.docx', 'example.pdf']
)
dataset = EvaluationDataset(goldens=goldens)
```

Полезно, когда нет продакшн-данных или эталонных датасетов.
### Запуск оценки

#### Способ 1: Pytest-интеграция

```python
# test_example.py
import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric
dataset = EvaluationDataset(goldens=[...])
# конвертация goldens → test cases...
@pytest.mark.parametrize("test_case", dataset.test_cases)
def test_chatbot(test_case: LLMTestCase):
    assert_test(test_case, [AnswerRelevancyMetric()])
```


```bash
deepeval test run test_example.py
```

**Параметры `assert_test()`:**
- `test_case` — обязательный
- `metrics` — обязательный (список)
- `run_async` — опциональный (по умолч. True)
#### Способ 2: Функция `evaluate()`
```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset
dataset = EvaluationDataset(goldens=[...])
evaluate(dataset, [AnswerRelevancyMetric()])
```

**Параметры `evaluate()`:**

|Параметр|Обязательный|По умолчанию|
|---|---|---|
|`test_cases`|✅|—|
|`metrics`|✅|—|
|`hyperparameters`|❌|`None`|
|`identifier`|❌|`None`|
|`async_config`|❌|`AsyncConfig()`|
|`display_config`|❌|`DisplayConfig()`|
|`error_config`|❌|`ErrorConfig()`|
|`cache_config`|❌|`CacheConfig()`|
### Оценка вложенных компонентов (Tracing)

Позволяет оценивать отдельные компоненты без изменения кода:
```python
from deepeval.tracing import observe, update_current_span
from deepeval.metrics import AnswerRelevancyMetric
@observe(metrics=[AnswerRelevancyMetric()])
def complete(query: str):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}]
    ).choices[0].message.content
    
    # Автоматическое логирование тест-кейса
    update_current_span(
        test_case=LLMTestCase(
            input=query,
            output=response
        )
    )
    return response
```

**Преимущества:**
- Разные метрики для разных компонентов
- Оценка нескольких компонентов одновременно
- Не нужно переписывать код для логирования

### Test Run

**Test run** — коллекция тест-кейсов, фиксирующая состояние LLM-приложения в момент времени.
- При входе в Confident AI → создается облачный отчёт
- Интегрируется в CI/CD через `.yaml` файлы


fdge4123fzfG