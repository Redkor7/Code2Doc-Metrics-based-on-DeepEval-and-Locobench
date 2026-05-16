from pydantic import BaseModel, Field
from typing import List

# --- (Schemas) ---

class FunctionExtraction(BaseModel):
    signature: str = Field(
        ..., 
        description="Extracted function signature (e.g., 'def my_func(a, b):')"
    )
    parameters_list: List[str] = Field(
        ..., 
        description="List of ONLY parameter names as plain strings (e.g., ['host', 'port']). DO NOT put objects or dictionaries here."
    )
    explanation: str = Field(
        ..., 
        description="The ENTIRE block of explanatory text from the documentation describing the function and ALL its parameters as a single string."
    )

class ExtractedFunctions(BaseModel):
    functions: List[FunctionExtraction]

class ParameterVerdict(BaseModel):
    signature: str
    verdict: str = Field(..., description="'yes' if ALL parameters are fully described (what they are and why they are needed), 'no' otherwise")
    reason: str = Field(..., description="Reason for the evaluation, explicitly stating which parameters are well-explained and which are missing")

class ParameterVerdicts(BaseModel):
    verdicts: List[ParameterVerdict]

class ParameterScoreReason(BaseModel):
    reason: str