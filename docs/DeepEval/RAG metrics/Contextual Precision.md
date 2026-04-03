### Описание
Метрика контекстуальной точности использует LLM-as-a-judge для измерения эффективности поиска в вашем конвейере RAG, оценивая, имеют ли узлы в вашем retrieval_context, которые имеют отношение к заданным входным данным, более высокий рейтинг, чем нерелевантные. контекстуальная метрика точности deepeval - это самоочевидная оценка LLM, то есть она выводит причину для получения оценки по метрике.

ContextualPrecisionMetric фокусируется на оценке повторной сортировки поискового устройства вашего RAG-конвейера путем оценки порядка ранжирования фрагментов текста в retrieval_context.

### Необходимые аргументы 
Чтобы использовать FaithfulnessMetric, вам нужно будет указать следующие аргументы при создании LLMTestCase: 
- input 
- actual_output 
- expected_output
- retrieval_context
### Пример использования:
```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric

# Replace this with the actual output from your LLM application
actual_output = "We offer a 30-day full refund at no extra cost."

# Replace this with the expected output of your RAG generator
expected_output = "You are eligible for a 30 day full refund at no extra cost."

# Replace this with the actual retrieved context from your RAG pipeline
retrieval_context = ["All customers are eligible for a 30 day full refund at no extra cost."]

metric = ContextualPrecisionMetric(
    threshold=0.7,
    model="gpt-4.1",
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
- evaluation_template: класс типа ContextualPrecisionTemplate, который позволяет вам переопределять запросы по умолчанию, используемые для вычисления показателя ContextualPrecisionTemplate. По умолчанию используется ContextualPrecisionTemplate от deepeval.
  
### Как происходит рассчет?

Оценка контекстуальной точности рассчитывается в соответствии со следующим уравнением:
$Contextual Precision = 1 / Number of Relevant Nodes * ∑ _{k=1} ^{n} ((Number of Relevant Nodes Up to Position k​ / k) × r​_k)$ 
- k - это (i+1)-й узел в retrieval_context , 
- n - длина retrieval_context, 
- r_k - двоичная релевантность для k-го узла в retrieval_context. rk = 1 для релевантных узлов, 0 в противном случае. 

ContextualPrecisionMetric сначала использует LLM, чтобы определить для каждого узла в retrieval_context, имеет ли он отношение к входным данным, основываясь на информации в expected_output, прежде чем вычислять взвешенную совокупную точность как показатель контекстуальной точности. Взвешенная кумулятивная точность (WCP) используется потому, что она: 
- Особое внимание уделяется лучшим результатам: WCP уделяет повышенное внимание релевантности результатов с самым высоким рейтингом. Этот акцент важен, поскольку LLM, как правило, уделяют больше внимания более ранним узлам в retrieval_context (что может вызвать галлюцинации в дальнейшем, если узлы ранжированы неправильно). 
- Упорядочение по значимости вознаграждается: WCP может обрабатывать данные различной степени значимости (например, "очень важные", "в некоторой степени важные", "нерелевантные"). Это отличается от таких показателей, как точность, которые рассматривают все найденные узлы как одинаково важные. Более высокий показатель контекстуальной точности означает большую способность поисковой системы корректно ранжировать соответствующие узлы выше в retrieval_context.