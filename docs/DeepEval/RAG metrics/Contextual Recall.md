### Описание
Метрика Контекстуальное воспроизведение использует LLM-as-a-judge для измерения качества поиска в вашем конвейере RAG путем оценки степени совпадения retrieval_context с ожидаемым результатом. В данной метрике выводится пояснение, почему была поставллена именно эта оценка.

### Необходимые аргументы 
Чтобы использовать ContextualRecallMetric, вам нужно будет указать следующие аргументы при создании LLMTestCase: 
- input 
- actual_output 
- expected_output
- retrieval_context
### Пример использования:
```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric

# Replace this with the actual output from your LLM application
actual_output = "We offer a 30-day full refund at no extra cost."

# Replace this with the expected output from your RAG generator
expected_output = "You are eligible for a 30 day full refund at no extra cost."

# Replace this with the actual retrieved context from your RAG pipeline
retrieval_context = ["All customers are eligible for a 30 day full refund at no extra cost."]

metric = ContextualRecallMetric(
    threshold=0.7,
    model="gpt-4",
    include_reason=True
)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=retrieval_context
)

# To run metric as a standalone
# metric.measure(test_case)
# print(metric.score, metric.reason)

evaluate(test_cases=[test_case], metrics=[metric])
```

При создании метрики есть 7 необязательных параметров: 
- threshold: значение с плавающей точкой, представляющее минимальный порог прохождения, по умолчанию равно 0,5. 
- model: строка, указывающая, какую из GPT-моделей OpenAI использовать, ИЛИ любую пользовательскую LLM-модель типа DeepEvalBaseLLM. По умолчанию используется 'gpt-4.1'. 
- include_reason: логическое значение, которое, если ему присвоено значение True, будет содержать причину для оценки. По умолчанию установлено значение True. 
- strict_mode: логическое значение, которое, если ему присвоено значение True, устанавливает двоичную оценку по метрике: 1 для совершенства, 0 в противном случае. Оно также переопределяет текущее пороговое значение и устанавливает его равным 1. По умолчанию используется значение False. 
- async_mode: логическое значение, которое, если ему присвоено значение True, включает параллельное выполнение в методе measure(). По умолчанию установлено значение True. 
- verbose_mode: логическое значение, которое, если ему присвоено значение True, выводит промежуточные шаги, используемые для вычисления указанного показателя, на консоль, как описано в разделе "Как это вычисляется". По умолчанию установлено значение False. 
- evaluation_template: класс типа ContextualRecallTemplate, который позволяет вам переопределять запросы по умолчанию, используемые для вычисления показателя ContextualRecallTemplate. По умолчанию используется ContextualRecallTemplate от deepeval.
  
### Как происходит рассчет?

Оценка ContextualRecallMetric рассчитывается в соответствии со следующим уравнением:

`Contextual Recall=Number of Attributable Statements/Total Number of Statements`

ContextualRecallMetric сначала использует LLM для извлечения всех утверждений, сделанных в expected_output, прежде чем использовать тот же LLM для классификации того, может ли каждый утверждение быть отнесено к узлам в retrieval_context.

Более высокий показатель контекстуального запоминания означает большую способность поисковой системы собирать всю релевантную информацию из общего доступного набора релевантных данных в вашей базе знаний.