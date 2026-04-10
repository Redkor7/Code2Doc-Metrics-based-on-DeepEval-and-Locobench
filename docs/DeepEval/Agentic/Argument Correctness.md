### Описание
Показатель корректности аргументов - это агентский показатель LLM, который оценивает способность вашего агента LLM генерировать правильные аргументы для вызываемых им инструментов. Он рассчитывается путем определения правильности аргументов для каждого вызова инструмента на основе входных данных.
### Необходимые аргументы 
Чтобы использовать ArgumentCorrectnessMetric, вам нужно будет указать следующие аргументы при создании LLMTestCase: 
- input 
- actual_output 
- tools_called`
### Пример использования:
```python
from deepeval import evaluate
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

metric = ArgumentCorrectnessMetric(
    threshold=0.7,
    model="gpt-4",
    include_reason=True
)
test_case = LLMTestCase(
    input="When did Trump first raise tariffs?",
    actual_output="Trump first raised tariffs in 2018 during the U.S.-China trade war.",
    tools_called=[
        ToolCall(
            name="WebSearch Tool",
            description="Tool to search for information on the web.",
            input={"search_query": "Trump first raised tariffs year"}
        ),
        ToolCall(
            name="History FunFact Tool",
            description="Tool to provide a fun fact about the topic.",
            input={"topic": "Trump tariffs"}
        )
    ]
)

# To run metric as a standalone
# metric.measure(test_case)
# print(metric.score, metric.reason)

evaluate(test_cases=[test_case], metrics=[metric])
```

При создании метрики есть 6 необязательных параметров: 
- threshold: значение с плавающей запятой, представляющее минимальный порог прохождения, по умолчанию равно 0,5. 
- include_reason: логическое значение, которое, если ему присвоено значение True, будет содержать причину для оценки. По умолчанию установлено значение True. 
- strict_mode: логическое значение, которое, если ему присвоено значение True, устанавливает двоичную оценку по метрике: 1 для совершенства, 0 в противном случае. Оно также переопределяет текущее пороговое значение и устанавливает его равным 1. По умолчанию используется значение False. 
- verbose_mode: логическое значение, которое, если ему присвоено значение True, выводит промежуточные шаги, используемые для вычисления указанного показателя, на консоль, как описано в разделе "Как это вычисляется". По умолчанию установлено значение False. 
- async_mode: логическое значение, которое, если ему присвоено значение True, включает параллельное выполнение в методе measure(). По умолчанию установлено значение True. 
- model: строка, указывающая, какую из GPT-моделей OpenAI использовать, ИЛИ любую пользовательскую LLM-модель типа DeepEvalBaseLLM. По умолчанию установлено значение 'gpt-4.1'.

### Как происходит рассчет?
Показатель ArgumentCorrectnessMetric рассчитывается в соответствии со следующим уравнением:

Argument Correctness = Number of Correctly Generated Input Parameters​ / Total Number of Tool Calls

ArgumentCorrectnessMetric оценивает корректность аргументов (входных параметров) для каждого вызова инструмента на основе задачи, описанной во входных данных.