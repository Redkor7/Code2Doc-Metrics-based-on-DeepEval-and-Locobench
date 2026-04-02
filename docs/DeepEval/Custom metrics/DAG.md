## DAGMetric (Deep Acyclic Graph Metric) — что это?

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
  