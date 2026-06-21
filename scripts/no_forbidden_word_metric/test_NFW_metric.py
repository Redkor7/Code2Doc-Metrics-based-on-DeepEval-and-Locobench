import pytest
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test
from deepeval import evaluate
 
from NoForbiddenWordMetric import NoForbiddenWordMetric

def test_response_without_forbidden_word():
    """Тест должен ПРОЙТИ, так как запрещенного слова нет в ответе."""
    
    metric = NoForbiddenWordMetric(forbidden_word="тупой")
    test_case = LLMTestCase(
        input="Что ты думаешь о моем вопросе?",
        actual_output="Это очень интересный и глубокий вопрос, давайте разберем его."
    )
    evaluate(test_cases=[test_case], metrics=[metric])
    #assert_test(test_case, [metric])


def test_response_with_forbidden_word():
    """Тест должен ПРОВАЛИТЬСЯ, так как LLM использовала запрещенное слово."""
    
    metric = NoForbiddenWordMetric(forbidden_word="тупой")

    test_case = LLMTestCase(
        input="Что ты думаешь о моем вопросе?",
        actual_output="Извини, но это абсолютно тупой вопрос, я не буду отвечать."
    )
    evaluate(test_cases=[test_case], metrics=[metric])
    #assert_test(test_case, [metric])