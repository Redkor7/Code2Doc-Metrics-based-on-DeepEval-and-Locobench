from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

class NoForbiddenWordMetric(BaseMetric):
    def __init__(self, forbidden_word: str, threshold: float = 1.0):
        self.forbidden_word = forbidden_word.lower()
        self.threshold = threshold
        self._required_params = [LLMTestCaseParams.ACTUAL_OUTPUT] 
        self.async_mode = False

    def measure(self, test_case: LLMTestCase) -> float:
        actual_output = test_case.actual_output.lower()
        
        if self.forbidden_word in actual_output:
            self.score = 0.0
            self.reason = f"Найдено запрещенное слово: {self.forbidden_word}"
        else:
            self.score = 1.0
            self.reason = "Запрещенных слов не найдено."
            
        self.success = self.is_successful()
        
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "No Forbidden Word Metric"