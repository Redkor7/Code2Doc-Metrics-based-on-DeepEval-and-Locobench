### Описание
В метрике Достоверность используется LLM-a-a-judge для оценки качества генератора вашего конвейера RAG путем оценки того, соответствует ли actual_output содержимому вашего retrieval_context. показатель верности deepeval - это самоочевидная оценка LLM, то есть она выводит причину получения оценки по этому показателю.

### Необходимые аргументы 
Чтобы использовать FaithfulnessMetric, вам нужно будет указать следующие аргументы при создании LLMTestCase: 
- input 
- actual_output 
- retrieval_context
  
### Пример использования:

```python
from deepeval import evaluate  
from deepeval.test_case import LLMTestCase  
from deepeval.metrics import FaithfulnessMetric  
  
# Replace this with the actual output from your LLM application  
actual_output = "We offer a 30-day full refund at no extra cost."  
  
# Replace this with the actual retrieved context from your RAG pipeline  
retrieval_context = ["All customers are eligible for a 30 day full refund at no extra cost."]  
  
metric = FaithfulnessMetric(  
threshold=0.7,  
model="gpt-4.1",  
include_reason=True  
)  
test_case = LLMTestCase(  
input="What if these shoes don't fit?",  
actual_output=actual_output,  
retrieval_context=retrieval_context  
)  
  
# To run metric as a standalone  
# metric.measure(test_case)  
# print(metric.score, metric.reason)  
  
evaluate(test_cases=[test_case], metrics=[metric])
```

При создании метрики используются восемь необязательных параметров: 
- threshold: значение с плавающей точкой, представляющее минимальный порог прохождения, по умолчанию равно 0,5. 
- model: строка, указывающая, какую из GPT-моделей OpenAI использовать, ИЛИ любую пользовательскую LLM-модель типа DeepEvalBaseLLM. По умолчанию используется 'gpt-4.1'. 
- include_reason: логическое значение, которое, если ему присвоено значение True, будет содержать причину для оценки. По умолчанию установлено значение True. 
- strict_mode: логическое значение, которое, если ему присвоено значение True, устанавливает двоичную оценку по метрике: 1 для совершенства, 0 в противном случае. Оно также переопределяет текущее пороговое значение и устанавливает его равным 1. По умолчанию используется значение False. 
- async_mode: логическое значение, которое, если ему присвоено значение True, включает параллельное выполнение в методе measure(). По умолчанию установлено значение True. 
- verbose_mode: логическое значение, которое, если ему присвоено значение True, выводит промежуточные шаги, используемые для вычисления указанного показателя, на консоль, как описано в разделе "Как это вычисляется". По умолчанию установлено значение False. 
- truths_extraction_limit: значение int, которое при установке определяет максимальное количество фактических данных для извлечения из retrieval_context. Полученные данные будут использованы для определения степени соответствия фактическим данным и будут упорядочены по степени важности в соответствии с вашей моделью оценки. По умолчанию задано значение "Нет". 
- penalize_ambiguous_claims: логическое значение, которое, если ему присвоено значение True, не будет учитывать неоднозначные утверждения как достоверные. По умолчанию установлено значение False. 
- evaluation_template: класс типа FaithfulnessTemplate, который позволяет вам переопределять запросы по умолчанию, используемые для вычисления оценки FaithfulnessMetric. По умолчанию используется FaithfulnessTemplate от deepeval.
  
### Как происходит рассчет?

Показатель FaithfulnessMetric рассчитывается в соответствии со следующим уравнением: `Достоверность = Количество правдивых заявлений / Общее количество заявлений `

FaithfulnessMetric сначала использует LLM для извлечения всех утверждений, сделанных в actual_output, а затем использует тот же LLM для определения того, является ли каждое утверждение правдивым, на основе фактов, представленных в retrieval_context. Утверждение считается правдивым, если оно не противоречит никаким фактам, представленным в retrieval_context.