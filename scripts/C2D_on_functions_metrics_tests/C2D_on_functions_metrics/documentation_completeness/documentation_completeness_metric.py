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
    RequiredDocElement, 
    RequiredDocElements, 
    CompletenessVerdict, 
    CompletenessVerdicts, 
    CompletenessScoreReason
)
from .template import DocumentationCompletenessTemplate


class DocumentationCompletenessMetric(BaseMetric):
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
        self.evaluation_template = DocumentationCompletenessTemplate()

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

                # Step 1: Create a checklist based on the code
                self.required_elements = self._generate_requirements(input_code)
                
                if not self.required_elements:
                    self.score = 1.0
                    self.reason = "No documentation elements found (possibly the code is empty)."
                    self.success = True
                    return self.score

                # Step 2: Compare the generated documentation with the checklist
                self.verdicts = self._generate_verdicts(self.required_elements, actual_output)
                
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Required Elements Checklist:\n{prettify_list([e.model_dump() for e in self.required_elements])}",
                        f"Completeness Verdicts:\n{prettify_list([v.model_dump() for v in self.verdicts])}",
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

            self.required_elements = await self._a_generate_requirements(input_code)
            if not self.required_elements:
                self.score = 1.0
                self.reason = "The source code requires no specific documentation elements."
                self.success = True
                return self.score

            self.verdicts = await self._a_generate_verdicts(self.required_elements, actual_output)
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Required Elements Checklist:\n{prettify_list([e.model_dump() for e in self.required_elements])}",
                    f"Completeness Verdicts:\n{prettify_list([v.model_dump() for v in self.verdicts])}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score
    
    def _generate_requirements(self, input_code: str) -> List[RequiredDocElement]:
        prompt = self.evaluation_template.extract_requirements(input_code)
        return generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=RequiredDocElements,
            extract_schema=lambda x: x.elements, extract_json=lambda d: [RequiredDocElement(**e) for e in d["elements"]]
        )

    def _generate_verdicts(self, elements: List[RequiredDocElement], actual_output: str) -> List[CompletenessVerdict]:
        prompt = self.evaluation_template.generate_verdicts(elements, actual_output)
        return generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=CompletenessVerdicts,
            extract_schema=lambda x: x.verdicts, extract_json=lambda d: [CompletenessVerdict(**v) for v in d["verdicts"]]
        )

    def _generate_reason(self) -> str:
        if not self.include_reason: return None
        missing_reasons = [f"Missing '{v.element_name}': {v.reason}" for v in self.verdicts if v.verdict.strip().lower() == "no"]
        if not missing_reasons: return "Documentation fully describes all required elements of the source code."
        prompt = self.evaluation_template.generate_reason(missing_reasons, format(self.score, ".2f"))
        return generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=CompletenessScoreReason,
            extract_schema=lambda x: x.reason, extract_json=lambda d: d["reason"]
        )

    async def _a_generate_requirements(self, input_code: str) -> List[RequiredDocElement]:
        prompt = self.evaluation_template.extract_requirements(input_code)
        return await a_generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=RequiredDocElements,
            extract_schema=lambda x: x.elements, extract_json=lambda d: [RequiredDocElement(**e) for e in d["elements"]]
        )

    async def _a_generate_verdicts(self, elements: List[RequiredDocElement], actual_output: str) -> List[CompletenessVerdict]:
        prompt = self.evaluation_template.generate_verdicts(elements, actual_output)
        return await a_generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=CompletenessVerdicts,
            extract_schema=lambda x: x.verdicts, extract_json=lambda d: [CompletenessVerdict(**v) for v in d["verdicts"]]
        )

    async def _a_generate_reason(self) -> str:
        if not self.include_reason: return None
        missing_reasons = [f"Missing '{v.element_name}': {v.reason}" for v in self.verdicts if v.verdict.strip().lower() == "no"]
        if not missing_reasons: return "Documentation fully describes all required elements of the source code."
        prompt = self.evaluation_template.generate_reason(missing_reasons, format(self.score, ".2f"))
        return await a_generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=CompletenessScoreReason,
            extract_schema=lambda x: x.reason, extract_json=lambda d: d["reason"]
        )

    def _calculate_score(self):
        if len(self.verdicts) == 0: return 1.0
        # Score = number of documented elements / total number of required elements
        covered_count = sum(1 for v in self.verdicts if v.verdict.strip().lower() != "no")
        score = covered_count / len(self.verdicts)
        return 0 if self.strict_mode and score < self.threshold else score

    def is_successful(self) -> bool:
        try:
            self.success = self.score >= self.threshold
        except TypeError:
            self.success = False
        return self.success

    @property
    def __name__(self):
        return "Documentation Completeness Metric"