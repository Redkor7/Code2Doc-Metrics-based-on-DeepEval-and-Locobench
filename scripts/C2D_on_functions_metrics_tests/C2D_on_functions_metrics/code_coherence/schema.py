from pydantic import BaseModel, Field
from typing import List

# --- (Schemas for Code Coherence / Developer Utility) ---

class CoherenceVerdict(BaseModel):
    aspect: str = Field(
        ..., 
        description="The specific aspect of coherence being evaluated (e.g., 'Intent', 'Abstraction', 'Clarity')."
    )
    verdict: str = Field(
        ..., 
        description="'yes' if the documentation successfully meets this aspect of quality, 'no' if it fails or is poorly executed."
    )
    reason: str = Field(
        ..., 
        description="Detailed technical justification for the verdict, quoting specific parts of the documentation or code."
    )

class CoherenceVerdicts(BaseModel):
    verdicts: List[CoherenceVerdict]

class CoherenceScoreReason(BaseModel):
    reason: str