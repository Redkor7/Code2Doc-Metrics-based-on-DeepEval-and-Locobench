from pydantic import BaseModel, Field
from typing import List

# --- (Schemas for Documentation Completeness) ---

class RequiredDocElement(BaseModel):
    element_name: str = Field(
        ..., 
        description="The specific element that needs documentation (e.g., 'General Purpose', 'Parameter: max_retries', 'Return Value', 'Exception: ValueError')."
    )
    justification: str = Field(
        ..., 
        description="Brief reason why this element exists in the code and must be documented."
    )

class RequiredDocElements(BaseModel):
    elements: List[RequiredDocElement]

class CompletenessVerdict(BaseModel):
    element_name: str
    verdict: str = Field(..., description="'yes' if the element is explicitly and adequately explained in the documentation, 'no' if it is missing or vaguely mentioned.")
    reason: str = Field(..., description="Explanation of the verdict. If 'yes', quote the part of the documentation. If 'no', explain what is missing.")

class CompletenessVerdicts(BaseModel):
    verdicts: List[CompletenessVerdict]

class CompletenessScoreReason(BaseModel):
    reason: str