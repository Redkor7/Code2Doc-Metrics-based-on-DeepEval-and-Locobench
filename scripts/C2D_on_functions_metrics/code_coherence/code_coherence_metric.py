import asyncio
from typing import List, Optional, Union

from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    generate_with_schema_and_extract,
    a_generate_with_schema_and_extract,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator

from .schema import (
    CoherenceVerdict, 
    CoherenceVerdicts, 
    CoherenceScoreReason
)
from .template import CodeCoherenceTemplate


class CodeCoherenceMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.evaluation_template = CodeCoherenceTemplate()

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        check_llm_test_case_params(
            test_case, self._required_params, None, None, self, self.model, getattr(test_case, "multimodal", False)
        )
        self.evaluation_cost = 0 if self.using_native_model else None
        
        with metric_progress_indicator(self, _show_indicator=_show_indicator, _in_component=_in_component):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(self.a_measure(test_case, False, _in_component, _log_metric_to_confident))
            else:
                input_code = test_case.input
                actual_output = test_case.actual_output

                self.verdicts = self._generate_verdicts(input_code, actual_output)
                
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Coherence Verdicts:\n{prettify_list([v.model_dump() for v in self.verdicts])}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
            return self.score

    async def a_measure(self, test_case: LLMTestCase, _show_indicator: bool = True, _in_component: bool = False, _log_metric_to_confident: bool = True) -> float:
        check_llm_test_case_params(
            test_case, self._required_params, None, None, self, self.model, getattr(test_case, "multimodal", False)
        )
        self.evaluation_cost = 0 if self.using_native_model else None
        
        with metric_progress_indicator(self, async_mode=True, _show_indicator=_show_indicator, _in_component=_in_component):
            input_code = test_case.input
            actual_output = test_case.actual_output

            self.verdicts = await self._a_generate_verdicts(input_code, actual_output)
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Coherence Verdicts:\n{prettify_list([v.model_dump() for v in self.verdicts])}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    def _generate_verdicts(self, input_code: str, actual_output: str) -> List[CoherenceVerdict]:
        prompt = self.evaluation_template.generate_verdicts(input_code, actual_output)
        return generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=CoherenceVerdicts,
            extract_schema=lambda x: x.verdicts, extract_json=lambda d: [CoherenceVerdict(**v) for v in d["verdicts"]]
        )

    def _generate_reason(self) -> str:
        if not self.include_reason: return None
        failed_reasons = [f"Failed '{v.aspect}': {v.reason}" for v in self.verdicts if v.verdict.strip().lower() == "no"]
        if not failed_reasons: return "The documentation is highly coherent, explains the intent, and is written in excellent technical language."
        prompt = self.evaluation_template.generate_reason(failed_reasons, format(self.score, ".2f"))
        return generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=CoherenceScoreReason,
            extract_schema=lambda x: x.reason, extract_json=lambda d: d["reason"]
        )

    async def _a_generate_verdicts(self, input_code: str, actual_output: str) -> List[CoherenceVerdict]:
        prompt = self.evaluation_template.generate_verdicts(input_code, actual_output)
        return await a_generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=CoherenceVerdicts,
            extract_schema=lambda x: x.verdicts, extract_json=lambda d: [CoherenceVerdict(**v) for v in d["verdicts"]]
        )

    async def _a_generate_reason(self) -> str:
        if not self.include_reason: return None
        failed_reasons = [f"Failed '{v.aspect}': {v.reason}" for v in self.verdicts if v.verdict.strip().lower() == "no"]
        if not failed_reasons: return "The documentation is highly coherent, explains the intent, and is written in excellent technical language."
        prompt = self.evaluation_template.generate_reason(failed_reasons, format(self.score, ".2f"))
        return await a_generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=CoherenceScoreReason,
            extract_schema=lambda x: x.reason, extract_json=lambda d: d["reason"]
        )

    def _calculate_score(self):
        if len(self.verdicts) == 0: return 1.0
        passed_count = sum(1 for v in self.verdicts if v.verdict.strip().lower() != "no")
        score = passed_count / len(self.verdicts)
        return 0 if self.strict_mode and score < self.threshold else score

    def is_successful(self) -> bool:
        try:
            self.success = self.score >= self.threshold
        except TypeError:
            self.success = False
        return self.success

    @property
    def __name__(self):
        return "Code Coherence Metric"