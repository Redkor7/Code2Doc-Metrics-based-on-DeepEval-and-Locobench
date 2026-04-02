from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4.1",
    include_reason=True
)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the output from your LLM app
    actual_output="We offer a 30-day full refund at no extra cost."
)

# To run metric as a standalone
# metric.measure(test_case)
# print(metric.score, metric.reason)

evaluate(test_cases=[test_case], metrics=[metric])