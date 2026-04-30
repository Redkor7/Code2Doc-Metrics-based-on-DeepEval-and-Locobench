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

from schema import (
    ExpectedFacts,
    FactVerdict,
    CompletenessVerdicts,
    CompletenessScoreReason
)
from template import SemanticCompletenessTemplate


class SemanticCompletenessMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
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
        self.evaluation_template = SemanticCompletenessTemplate()

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
                actual_output = test_case.actual_output
                expected_output = test_case.expected_output

                # 1. extract a checklist of facts from the standard
                self.expected_facts = self._extract_facts(expected_output)
                
                if not self.expected_facts:
                    self.score = 1.0
                    self.reason = "In the reference document, no key facts were found for evaluation."
                    self.success = True
                    return self.score

                # 2. Checking with the actual response
                self.verdicts = self._evaluate_facts(actual_output, self.expected_facts)
                
                # 3. Calculate the result and generate a reason
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Expected Facts Extracted:\n{prettify_list(self.expected_facts)}",
                        f"Verdicts:\n{prettify_list([v.dict() for v in self.verdicts])}",
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
            actual_output = test_case.actual_output
            expected_output = test_case.expected_output

            self.expected_facts = await self._a_extract_facts(expected_output)
            
            if not self.expected_facts:
                self.score = 1.0
                self.reason = "In the reference document, no key facts were found for evaluation."
                self.success = True
                return self.score

            self.verdicts = await self._a_evaluate_facts(actual_output, self.expected_facts)
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Expected Facts Extracted:\n{prettify_list(self.expected_facts)}",
                    f"Verdicts:\n{prettify_list([v.dict() for v in self.verdicts])}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    
    def _extract_facts(self, expected_output: str) -> List[str]:
        prompt = self.evaluation_template.extract_facts(expected_output)
        return generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=ExpectedFacts,
            extract_schema=lambda x: x.facts, extract_json=lambda d: d["facts"]
        )

    def _evaluate_facts(self, actual_output: str, expected_facts: List[str]) -> List[FactVerdict]:
        prompt = self.evaluation_template.evaluate_facts(actual_output, expected_facts)
        return generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=CompletenessVerdicts,
            extract_schema=lambda x: x.verdicts, extract_json=lambda d: [FactVerdict(**v) for v in d["verdicts"]]
        )

    def _generate_reason(self) -> str:
        if not self.include_reason: return None
        # Collecting reasons only for missing facts (no)
        missing_facts = [f"Fact: {v.fact}\nReason: {v.reason}" for v in self.verdicts if v.verdict.strip().lower() == "no"]
        
        if not missing_facts: 
            return "The documentation contains all the necessary facts from the reference."
            
        prompt = self.evaluation_template.generate_reason(missing_facts, format(self.score, ".2f"))
        return generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=CompletenessScoreReason,
            extract_schema=lambda x: x.reason, extract_json=lambda d: d["reason"]
        )

    # --- Asynchronous LLM call methods ---
    
    async def _a_extract_facts(self, expected_output: str) -> List[str]:
        prompt = self.evaluation_template.extract_facts(expected_output)
        return await a_generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=ExpectedFacts,
            extract_schema=lambda x: x.facts, extract_json=lambda d: d["facts"]
        )

    async def _a_evaluate_facts(self, actual_output: str, expected_facts: List[str]) -> List[FactVerdict]:
        prompt = self.evaluation_template.evaluate_facts(actual_output, expected_facts)
        return await a_generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=CompletenessVerdicts,
            extract_schema=lambda x: x.verdicts, extract_json=lambda d: [FactVerdict(**v) for v in d["verdicts"]]
        )

    async def _a_generate_reason(self) -> str:
        if not self.include_reason: return None
        missing_facts = [f"Fact: {v.fact}\nReason: {v.reason}" for v in self.verdicts if v.verdict.strip().lower() == "no"]
        if not missing_facts: 
            return "The documentation contains all the necessary facts from the reference."
            
        prompt = self.evaluation_template.generate_reason(missing_facts, format(self.score, ".2f"))
        return await a_generate_with_schema_and_extract(
            metric=self, prompt=prompt, schema_cls=CompletenessScoreReason,
            extract_schema=lambda x: x.reason, extract_json=lambda d: d["reason"]
        )

    # --- Logic and Mathematics ---

    def _calculate_score(self):
        if len(self.verdicts) == 0: return 1.0

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
        return "Semantic Completeness Metric"