from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric

# 1. Decorate your app
@observe()
def llm_app(input: str):
  # 2. Decorate components with metrics you wish to evaluate or debug
  @observe(metrics=[AnswerRelevancyMetric()])
  def inner_component():
      # 3. Create test case at runtime
      update_current_span(test_case=LLMTestCase(input="Why is the blue sky?", actual_output="You mean why is the sky blue?"))

  return inner_component()

# 4. Create dataset
dataset = EvaluationDataset(goldens=[Golden(input="Test input")])

# 5. Loop through dataset
for golden in dataset.evals_iterator():
  # 6. Call LLM app
  llm_app(golden.input)