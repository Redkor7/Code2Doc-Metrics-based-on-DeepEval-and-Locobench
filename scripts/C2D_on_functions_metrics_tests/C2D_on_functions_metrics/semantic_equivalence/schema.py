from pydantic import BaseModel, Field
from typing import List

# --- (Schemas for Semantic Equivalence) ---

class SemanticClaim(BaseModel):
    claim: str = Field(
        ..., 
        description="A single, atomic factual claim extracted from the expected documentation (e.g., 'The function returns an integer', 'It handles file descriptor allocation')."
    )

class ExtractedSemanticClaims(BaseModel):
    claims: List[SemanticClaim]

class SemanticVerdict(BaseModel):
    claim: str
    verdict: str = Field(..., description="'yes' if the claim is fully supported by the actual output, 'no' otherwise")
    reason: str = Field(..., description="Explanation of why the claim is or is not supported by the actual output. If 'yes', briefly state the matching phrase.")

class SemanticVerdicts(BaseModel):
    verdicts: List[SemanticVerdict]

class SemanticScoreReason(BaseModel):
    reason: str