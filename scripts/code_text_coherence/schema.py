from pydantic import BaseModel, Field
from typing import List, Optional

# --- (Schemas) ---

class CodeExplanationPair(BaseModel):
    code: str = Field(..., description="Extracted code block")
    explanation: str = Field(..., description="Text from the documentation that explains this code block")

class ExtractedPairs(BaseModel):
    pairs: List[CodeExplanationPair]

class CoherenceVerdict(BaseModel):
    code: str
    verdict: str = Field(..., description="'yes' if the code and explanation are coherent, 'no' otherwise")
    reason: str = Field(..., description="Reason for the evaluation")

class CoherenceVerdicts(BaseModel):
    verdicts: List[CoherenceVerdict]

class CoherenceScoreReason(BaseModel):
    reason: str