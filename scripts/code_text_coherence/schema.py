from pydantic import BaseModel, Field
from typing import List, Optional

# --- (Schemas) ---

class CodeExplanationPair(BaseModel):
    code: str = Field(..., description="Извлеченный блок кода")
    explanation: str = Field(..., description="Текст из документации, который поясняет этот блок кода")

class ExtractedPairs(BaseModel):
    pairs: List[CodeExplanationPair]

class CoherenceVerdict(BaseModel):
    code: str
    verdict: str = Field(..., description="'yes' если код и описание связны, 'no' если нет")
    reason: str = Field(..., description="Причина оценки")

class CoherenceVerdicts(BaseModel):
    verdicts: List[CoherenceVerdict]

class CoherenceScoreReason(BaseModel):
    reason: str