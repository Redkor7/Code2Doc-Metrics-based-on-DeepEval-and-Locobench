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


┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Test Cases    │────▶│    Metrics      │────▶│   Evaluation   │
│   (что тестируем)│     │   (чем оцениваем)│     │    Dataset     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
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

## G-Eval — что это?

**G-Eval** — фреймворк для оценки LLM-ответов с использованием **LLM-as-a-judge** и **цепочки рассуждений (Chain-of-Thoughts, CoT)**. Позволяет создавать кастомные метрики под любые критерии.

---

### Как работает алгоритм G-Eval?

#### Двухэтапный процесс:

1. **Генерация шагов оценки (CoT)**
    - LLM получает критерии и параметры
    - Генерирует последовательность шагов для оценки

2. **Вычисление оценки**
    - Создается промпт с шагами и данными тест-кейса
    - LLM возвращает оценку и обоснование
    - **Нормализация** через вероятности токенов (weighted summation)

### Структура класса GEval

#### Основные параметры конструктора:


```python
class GEval(BaseMetric):
    def __init__(
        self,
        name: str,                                    # Имя метрики
        evaluation_params: List[LLMTestCaseParams],   # Какие поля использовать
        criteria: Optional[str] = None,               # Критерии (текст)
        evaluation_steps: Optional[List[str]] = None, # Шаги оценки
        rubric: Optional[List[Rubric]] = None,        # Шкала оценок
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,  # Модель
        threshold: float = 0.5,                       # Порог успеха
        top_logprobs: int = 20,                       # Для нормализации
        async_mode: bool = True,                      # Асинхронность
        strict_mode: bool = False,                    # Бинарный режим
        verbose_mode: bool = False,                   # Логирование
        evaluation_template: Type[GEvalTemplate] = GEvalTemplate,  # Кастомный шаблон
    ):

```

### Ключевые методы

#### 1. `measure()` — основной метод оценки
```python
def measure(
    self,
    test_case: LLMTestCase,
    _show_indicator: bool = True,
    _in_component: bool = False,
    _log_metric_to_confident: bool = True,
    _additional_context: Optional[str] = None,
) -> float:
```


**Этапы выполнения:**
```python
# 1. Проверка параметров
check_llm_test_case_params(test_case, self.evaluation_params, ...)
# 2. Генерация шагов (если не заданы)
self.evaluation_steps = self._generate_evaluation_steps(multimodal)
# 3. Оценка
g_score, reason = self._evaluate(test_case, ...)
# 4. Нормализация score из диапазона [min, max] в [0, 1]
self.score = (float(g_score) - self.score_range[0]) / self.score_range_span
# 5. Проверка успеха
self.success = self.score >= self.threshold
```

#### 2. `_generate_evaluation_steps()` — генерация CoT

```python
def _generate_evaluation_steps(self, multimodal: bool) -> List[str]:
    if self.evaluation_steps:
        return self.evaluation_steps  # Используем готовые
    
    # Создаем промпт из шаблона
    prompt = self.evaluation_template.generate_evaluation_steps(
        criteria=self.criteria,
        parameters=construct_g_eval_params_string(self.evaluation_params),
        multimodal=multimodal,
    )
    
    # Извлекаем JSON с шагами
    return generate_with_schema_and_extract(
        metric=self,
        prompt=prompt,
        schema_cls=gschema.Steps,        # Pydantic схема
        extract_schema=lambda s: s.steps,
        extract_json=lambda d: d["steps"],
    )
```
#### 3. `_evaluate()` — вычисление оценки

```python
def _evaluate(self, test_case: LLMTestCase, multimodal: bool, ...):
    # Формируем контент тест-кейса
    test_case_content = construct_test_case_string(
        self.evaluation_params, test_case
    )
    
    # Создаем промпт для оценки
    if not self.strict_mode:
        prompt = self.evaluation_template.generate_evaluation_results(
            evaluation_steps=number_evaluation_steps(self.evaluation_steps),
            test_case_content=test_case_content,
            parameters=g_eval_params_str,
            rubric=format_rubrics(self.rubric),
            score_range=self.score_range,
            _additional_context=_additional_context,
            multimodal=multimodal,
        )
    else:
        prompt = self.evaluation_template.generate_strict_evaluation_results(...)
    
    try:
        # Пытаемся получить логиты для нормализации
        res, cost = self.model.generate_raw_response(
            prompt, top_logprobs=self.top_logprobs
        )
        
        data = trimAndLoadJson(res.choices[0].message.content, self)
        reason = data["reason"]
        score = data["score"]
        
        if not self.strict_mode:
            # Нормализация через weighted summation
            weighted_summed_score = calculate_weighted_summed_score(score, res)
            return weighted_summed_score, reason
        
        return score, reason
        
    except AttributeError:
        # Если модель не поддерживает logprobs — обычный JSON
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=gschema.ReasonScore,
            extract_schema=lambda s: (s.score, s.reason),
            extract_json=lambda d: (d["score"], d["reason"]),
        )
```
### Нормализация оценки через logprobs

#### Как это работает:
```python
def calculate_weighted_summed_score(score, response):
    """Нормализация через вероятности токенов"""
    # Исходный score — число от 0 до 10
    # Получаем вероятности каждого токена из response
    # Вычисляем взвешенную сумму: sum(вероятность * значение_токена)
    # Это уменьшает bias LLM при выставлении оценок
```
### Rubric — ограничение диапазона оценок
```python
from deepeval.metrics.g_eval import Rubric
rubric = [
    Rubric(score_range=(0, 2), expected_outcome="Фактически неверно"),
    Rubric(score_range=(3, 6), expected_outcome="В основном верно"),
    Rubric(score_range=(7, 9), expected_outcome="Верно, но есть мелкие ошибки"),
    Rubric(score_range=(10, 10), expected_outcome="100% верно"),
]
# Диапазон будет: score_range = (0, 10), score_range_span = 10
```
### Кастомные шаблоны (Template)
```python
from deepeval.metrics.g_eval import GEvalTemplate
import textwrap
class CustomGEvalTemplate(GEvalTemplate):
    @staticmethod
    def generate_evaluation_steps(parameters: str, criteria: str, multimodal: bool):
        return textwrap.dedent(f"""
            Ты эксперт по оценке {parameters}.
            
            Критерии: {criteria}
            
            Создай 3-4 шага для оценки.
            Верни JSON: {{"steps": ["шаг1", "шаг2", ...]}}
        """)
metric = GEval(
    name="Custom",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    criteria="Проверь качество ответа",
    evaluation_template=CustomGEvalTemplate
)
```

### Пример использования
```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
# Создаем метрику
correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Проверь, не противоречат ли факты в 'actual output' фактам в 'expected output'",
        "Сурово штрафуй за пропуск деталей",
        "Неоднозначные формулировки допустимы"
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ]
)
# Оценка
test_case = LLMTestCase(
    input="Question?",
    actual_output="Answer",
    expected_output="Expected answer"
)
correctness.measure(test_case)
print(f"Score: {correctness.score}")      # 0.0-1.0
print(f"Reason: {correctness.reason}")    # Обоснование
print(f"Success: {correctness.success}")  # score >= threshold


```
### Ключевые особенности реализации

|Особенность|Реализация|
|---|---|
|**CoT генерация**|`generate_with_schema_and_extract` + Pydantic схемы|
|**Нормализация**|Через `generate_raw_response` с `top_logprobs`|
|**Rubric**|Ограничивает диапазон, валидация через `validate_and_sort_rubrics`|
|**Асинхронность**|`async_mode` + `a_measure`|
|**Кастомные промпты**|Наследование `GEvalTemplate`|
|**Строгий режим**|`strict_mode` → бинарный результат (0/1)|
### Важные нюансы
1. **Нельзя одновременно указывать** `criteria` и `evaluation_steps`
2. **Только нужные параметры** в `evaluation_params` — лишние снижают точность
3. **logprobs** нужны для нормализации, но не все модели поддерживают
4. **Rubric** не должен иметь пересекающихся диапазонов
   
   
##  DAGMetric (Deep Acyclic Graph Metric) — что это?

**DAGMetric** — это гибкая кастомная метрика для оценки LLM-приложений. Она позволяет создавать **детерминированные деревья решений**, где каждый шаг оценивается с помощью LLM-as-a-judge.
### Как работает?

Вы строите **ациклический граф решений (DAG)**, состоящий из узлов разных типов:
- **TaskNode** — обрабатывает данные (например, извлекает заголовки из текста)
- **BinaryJudgementNode** — отвечает на вопрос "да/нет"
- **NonBinaryJudgementNode** — допускает несколько вариантов ответа
- **VerdictNode** — листовой узел, присваивает итоговую оценку

Граф выполняется сверху вниз, выбирая путь на основе результатов каждого узла.

### Как создать DAGMetric?
```python
from deepeval.metrics import DAGMetric
from deepeval.dag import DeepAcyclicGraph
dag = DeepAcyclicGraph(root_nodes=[...])
metric = DAGMetric(name="MyMetric", dag=dag)
```

### Обязательные параметры:
- `name` — имя метрики
- `dag` — дерево решений
### Опциональные:
- `threshold` — порог прохождения (по умолч. 0.5)
- `model` — модель LLM
- `include_reason` — включать ли пояснение
- `strict_mode` — только 0 или 1
- `async_mode` — параллельное выполнение
- `verbose_mode` — вывод промежуточных шагов
### Пример узлов в коде
```python
from deepeval.metrics.dag import (
    TaskNode, BinaryJudgementNode, NonBinaryJudgementNode, VerdictNode
)
# Узел обработки
extract = TaskNode(
    instructions="Извлечь заголовки из actual_output",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    output_label="Заголовки"
)
# Бинарная проверка
binary = BinaryJudgementNode(
    criteria="Есть ли все три заголовка: intro, body, conclusion?",
    children=[
        VerdictNode(verdict=False, score=0),
        VerdictNode(verdict=True, child=...)
    ]
)
# Небинарная проверка
non_binary = NonBinaryJudgementNode(
    criteria="В каком порядке заголовки?",
    children=[
        VerdictNode(verdict="Правильный", score=10),
        VerdictNode(verdict="Два не на месте", score=4),
        VerdictNode(verdict="Все не на месте", score=2)
    ]
)
```
### Как считается результат?

1. Граф выполняется в топологическом порядке.
2. Каждый узел (кроме листьев) делает LLM-вывод.
3. В зависимости от ответа выбирается следующий узел.
4. Листовой `VerdictNode` возвращает итоговый `score` (0–10).
### Когда использовать?
- Когда критерии оценки сложные и многоступенчатые
- Когда нужна высокая детерминированность
- Когда вы хотите разбить оценку на независимые шаги
  
## RAG
Это архитектурный подход к построению LLM-приложений, при котором модель дополняется внешним поиском информации перед генерацией ответа.
### Проблема, которую решает RAG

У LLM есть фундаментальные ограничения:
- **Ограниченные знания** — модель знает только то, что было в training data (обычно до определенной даты)
- **Галлюцинации** — модель может уверенно выдумывать факты
- **Нет доступа к приватным данным** — документация компании, база знаний, личные файлы

RAG решает эти проблемы, подставляя в промпт **актуальные, релевантные данные** из внешнего источника.