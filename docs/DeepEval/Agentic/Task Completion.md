### Описание
Показатель выполнения задачи использует LLM-a-a-judge для оценки того, насколько эффективно агент LLM выполняет задачу. Завершение задачи - это самоочевидная оценка LLM, то есть она указывает причину получения оценки по этому показателю. Выполнение задачи анализирует полную трассировку вашего агента, чтобы определить успешность выполнения задачи, для чего требуется настроить трассировку.
### Пример использования:
```python
from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric

@observe()
def trip_planner_agent(input):
    destination = "Paris"
    days = 2

    @observe()
    def restaurant_finder(city):
        return ["Le Jules Verne", "Angelina Paris", "Septime"]

    @observe()
    def itinerary_generator(destination, days):
        return ["Eiffel Tower", "Louvre Museum", "Montmartre"][:days]

    itinerary = itinerary_generator(destination, days)
    restaurants = restaurant_finder(destination)

    return itinerary + restaurants


# Create dataset
dataset = EvaluationDataset(goldens=[Golden(input="This is a test query")])

# Initialize metric
task_completion = TaskCompletionMetric(threshold=0.7, model="gpt-4o")

# Loop through dataset
for golden in dataset.evals_iterator(metrics=[task_completion]):
    trip_planner_agent(golden.input)
```

При создании метрики есть 7 необязательных параметров: 
- threshold: значение с плавающей точкой, представляющее минимальный порог прохождения, по умолчанию равно 0,5. 
- task: строка, представляющая задачу, которая должна быть выполнена. Если задача не указана, она автоматически выводится из трассировки. По умолчанию установлено значение None 
- model: строка, указывающая, какую из GPT-моделей OpenAI использовать, или любую пользовательскую LLM-модель типа DeepEvalBaseLLM. По умолчанию установлено значение "gpt-4o". 
- include_reason: логическое значение, которое, если ему присвоено значение True, будет содержать причину для оценки. По умолчанию установлено значение True. 
- strict_mode: логическое значение, которое, если ему присвоено значение True, устанавливает двоичную оценку по метрике: 1 для совершенства, 0 в противном случае. Оно также переопределяет текущее пороговое значение и устанавливает его равным 1. По умолчанию используется значение False. 
- async_mode: логическое значение, которое, если ему присвоено значение True, включает параллельное выполнение в методе measure(). По умолчанию установлено значение True. 
- verbose_mode: логическое значение, которое, если ему присвоено значение True, выводит промежуточные шаги, используемые для вычисления указанного показателя, на консоль, как описано в разделе "Как это вычисляется". По умолчанию установлено значение False.
  
### Как происходит рассчет?

Оценка TaskCompletionMetric рассчитывается в соответствии со следующим уравнением:

`Task Completion Score = AlignmentScore(Task,Outcome)`

Задача(Task) и результат(Outcome) извлекаются из трассировки (или сквозного тестового примера) с использованием LLM. 
Оценка соответствия(AlignmentScore) измеряет, насколько хорошо результат согласуется с извлеченной (или предоставленной пользователем) задачей, по оценке LLM.