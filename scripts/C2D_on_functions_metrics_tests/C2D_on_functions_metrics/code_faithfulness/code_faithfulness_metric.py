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
    DocClaim, 
    ExtractedDocClaims, 
    FaithfulnessVerdict, 
    FaithfulnessVerdicts, 
    FaithfulnessScoreReason
)
from .template import CodeFaithfulnessTemplate


class CodeFaithfulnessMetric(BaseMetric):
    # We require the source code (input) and the generated documentation (actual_output)
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
        self.evaluation_template = CodeFaithfulnessTemplate()

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            getattr(test_case, "multimodal", False),
        )
        
        self.evaluation_cost = 0 if self.using_native_model else None
        
        with metric_progress_indicator(self, _show_indicator=_show_indicator, _in_component=_in_component):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(self.a_measure(test_case, False, _in_component, _log_metric_to_confident))
            else:
                input_code = test_case.input
                actual_output = test_case.actual_output

                # Step 1: Extract claims from the generated documentation
                self.claims = self._generate_claims(actual_output)
                
                if not self.claims:
                    self.score = 1.0
                    self.reason = "The documentation does not contain specific statements about the code."
                    self.success = True
                    return self.score

                # Step 2: Check the claims for faithfulness against the source code
                self.verdicts = self._generate_verdicts(self.claims, input_code)
                
                # Step 3: Calculate the score
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Extracted Claims from Docs:\n{prettify_list([c.model_dump() for c in self.claims])}",
                        f"Verdicts (Code Check):\n{prettify_list([v.model_dump() for v in self.verdicts])}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
            return self.score

    async def a_measure(self, test_case: LLMTestCase, _show_indicator: bool = True, _in_component: bool = False, _log_metric_to_confident: bool = True) -> float:
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            getattr(test_case, "multimodal", False),
        )
        self.evaluation_cost = 0 if self.using_native_model else None
        
        with metric_progress_indicator(self, async_mode=True, _show_indicator=_show_indicator, _in_component=_in_component):
            input_code = test_case.input
            actual_output = test_case.actual_output

            self.claims = await self._a_generate_claims(actual_output)
            if not self.claims:
                self.score = 1.0
                self.reason = "The documentation contains no specific claims to verify against the code."
                self.success = True
                return self.score

            self.verdicts = await self._a_generate_verdicts(self.claims, input_code)
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Extracted Claims from Docs:\n{prettify_list([c.model_dump() for c in self.claims])}",
                    f"Verdicts (Code Check):\n{prettify_list([v.model_dump() for v in self.verdicts])}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score
    
    def _generate_claims(self, actual_output: str) -> List[DocClaim]:
        prompt = self.evaluation_template.extract_claims(actual_output)
        return generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=ExtractedDocClaims,
            extract_schema=lambda x: x.claims, extract_json=lambda d: [DocClaim(**c) for c in d["claims"]]
        )

    def _generate_verdicts(self, claims: List[DocClaim], input_code: str) -> List[FaithfulnessVerdict]:
        prompt = self.evaluation_template.generate_verdicts(claims, input_code)
        return generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=FaithfulnessVerdicts,
            extract_schema=lambda x: x.verdicts, extract_json=lambda d: [FaithfulnessVerdict(**v) for v in d["verdicts"]]
        )

    def _generate_reason(self) -> str:
        if not self.include_reason: return None
        hallucinated_reasons = [v.reason for v in self.verdicts if v.verdict.strip().lower() == "no"]
        if not hallucinated_reasons: return "The documentation is fully faithful to the source code, no hallucinations detected."
        prompt = self.evaluation_template.generate_reason(hallucinated_reasons, format(self.score, ".2f"))
        return generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=FaithfulnessScoreReason,
            extract_schema=lambda x: x.reason, extract_json=lambda d: d["reason"]
        )

    async def _a_generate_claims(self, actual_output: str) -> List[DocClaim]:
        prompt = self.evaluation_template.extract_claims(actual_output)
        return await a_generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=ExtractedDocClaims,
            extract_schema=lambda x: x.claims, extract_json=lambda d: [DocClaim(**c) for c in d["claims"]]
        )

    async def _a_generate_verdicts(self, claims: List[DocClaim], input_code: str) -> List[FaithfulnessVerdict]:
        prompt = self.evaluation_template.generate_verdicts(claims, input_code)
        return await a_generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=FaithfulnessVerdicts,
            extract_schema=lambda x: x.verdicts, extract_json=lambda d: [FaithfulnessVerdict(**v) for v in d["verdicts"]]
        )

    async def _a_generate_reason(self) -> str:
        if not self.include_reason: return None
        hallucinated_reasons = [v.reason for v in self.verdicts if v.verdict.strip().lower() == "no"]
        if not hallucinated_reasons: return "The documentation is fully faithful to the source code, no hallucinations detected."
        prompt = self.evaluation_template.generate_reason(hallucinated_reasons, format(self.score, ".2f"))
        return await a_generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=FaithfulnessScoreReason,
            extract_schema=lambda x: x.reason, extract_json=lambda d: d["reason"]
        )

    def _calculate_score(self):
        if len(self.verdicts) == 0: return 1.0
        supported_count = sum(1 for v in self.verdicts if v.verdict.strip().lower() != "no")
        score = supported_count / len(self.verdicts)
        return 0 if self.strict_mode and score < self.threshold else score

    def is_successful(self) -> bool:
        try:
            self.success = self.score >= self.threshold
        except TypeError:
            self.success = False
        return self.success

    @property
    def __name__(self):
        return "Code Faithfulness Metric"