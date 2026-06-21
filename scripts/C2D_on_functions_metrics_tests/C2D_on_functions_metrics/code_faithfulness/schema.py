from pydantic import BaseModel, Field
from typing import List

# --- (Schemas for Code Faithfulness) ---

class DocClaim(BaseModel):
    claim: str = Field(
        ..., 
        description="A single, atomic factual claim extracted from the generated documentation (e.g., 'The function accepts a timeout parameter', 'It raises a ValueError if the input is negative')."
    )

class ExtractedDocClaims(BaseModel):
    claims: List[DocClaim]

class FaithfulnessVerdict(BaseModel):
    claim: str
    verdict: str = Field(..., description="'yes' if the claim can be logically inferred or proven strictly from the provided source code, 'no' if it is hallucinated, contradictory, or unprovable from the code.")
    reason: str = Field(..., description="Explanation of why the claim is supported or hallucinated. Refer to specific lines or logic in the source code.")

class FaithfulnessVerdicts(BaseModel):
    verdicts: List[FaithfulnessVerdict]

class FaithfulnessScoreReason(BaseModel):
    reason: str