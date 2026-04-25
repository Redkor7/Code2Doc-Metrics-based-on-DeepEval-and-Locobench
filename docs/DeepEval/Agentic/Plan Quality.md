### Описание
Показатель качества плана - это агентский показатель, который извлекает задачу и план из данных вашего агента, которые затем используются для оценки качества плана выполнения задачи. В данной метрике выводится пояснение, почему была поставллена именно эта оценка.
### Пример использования:
```python
from somewhere import llm
from deepeval.tracing import observe, update_current_trace
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import PlanQualityMetric
from deepeval.test_case import ToolCall


@observe
def tool_call(input):
    ...
    return [ToolCall(name="CheckWhether")]

@observe
def agent(input):
    tools = tool_call(input)
    output = llm(input, tools)
    update_current_trace(
        input=input,
        output=output,
        tools_called=tools
    )
    return output


# Create dataset
dataset = EvaluationDataset(goldens=[Golden(input="What's the weather like in SF?")])

# Initialize metric
metric = PlanQualityMetric(threshold=0.7, model="gpt-4o")

# Loop through dataset
for golden in dataset.evals_iterator(metrics=[metric]):
    agent(golden.input)
```

При создании метрики есть 6 необязательных параметров: 
- threshold: значение с плавающей точкой, представляющее минимальный порог прохождения, по умолчанию равно 0,5. 
- model: строка, указывающая, какую из GPT-моделей OpenAI использовать, или любую пользовательскую LLM-модель типа DeepEvalBaseLLM. По умолчанию установлено значение "gpt-4o". 
- include_reason: логическое значение, которое, если ему присвоено значение True, будет содержать причину для оценки. По умолчанию установлено значение True. 
- strict_mode: логическое значение, которое, если ему присвоено значение True, устанавливает двоичную оценку по метрике: 1 для совершенства, 0 в противном случае. Оно также переопределяет текущее пороговое значение и устанавливает его равным 1. По умолчанию используется значение False. 
- async_mode: логическое значение, которое, если ему присвоено значение True, включает параллельное выполнение в методе measure(). По умолчанию установлено значение True. 
- verbose_mode: логическое значение, которое, если ему присвоено значение True, выводит промежуточные шаги, используемые для вычисления указанного показателя, на консоль, как описано в разделе "Как это вычисляется". По умолчанию установлено значение False.
  
### Как происходит рассчет?

Показатель PlanQualityMetric рассчитывается путем выполнения следующих действий: 
- Извлекается задача из трассировки, это определяет цель пользователя или намерение агента и позволяет выполнять действия. 
- Извлекается план из трассировки, план извлекается из размышлений агента. Если в трассировке нет утверждений, которые четко определяют или подразумевают план, показатель по умолчанию оценивается как 1.

`Plan Quality Score=AlignmentScore(Task,Plan)``

AlignmentScore использует LLM для получения окончательной оценки со всей предварительно обработанной и извлеченной информацией, такой как план, задача и этапы выполнения.