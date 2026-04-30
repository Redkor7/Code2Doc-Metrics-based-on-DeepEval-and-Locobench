from typing import List
from pydantic import BaseModel, Field

class ExpectedFacts(BaseModel):
    facts: List[str] = Field(..., description="List of key facts extracted from the reference document")

class FactVerdict(BaseModel):
    fact: str = Field(..., description="The fact being evaluated")
    verdict: str = Field(..., description="'yes' if the fact is present in the generated response, 'no' if it's missing")
    reason: str = Field(..., description="Explanation for why the fact is considered present or absent")

class CompletenessVerdicts(BaseModel):
    verdicts: List[FactVerdict]

class CompletenessScoreReason(BaseModel):
    reason: str