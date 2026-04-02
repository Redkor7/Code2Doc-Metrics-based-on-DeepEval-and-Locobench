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