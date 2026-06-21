from typing import List
from .schema import DocClaim

class CodeFaithfulnessTemplate:
    @staticmethod
    def extract_claims(actual_output: str) -> str:
        return f"""
CRITICAL: OUTPUT ONLY VALID RAW JSON. DO NOT USE MARKDOWN BLOCKS (```json). DO NOT ADD ANY CONVERSATIONAL TEXT BEFORE OR AFTER THE JSON.    
    You are provided with generated technical documentation for a piece of code.
Your task is to break down this documentation into a list of atomic, factual claims. 
Extract claims about parameters, return types, exceptions raised, side effects, and core logic described in the text.

Generated Documentation:
{actual_output}

Return the result in JSON format, containing a 'claims' array where each element has a 'claim' string field.
"""

    @staticmethod
    def generate_verdicts(claims: List[DocClaim], input_code: str) -> str:
        claims_text = "\n".join([f"- {c.claim}" for c in claims])
        return f"""
    CRITICAL: OUTPUT ONLY VALID RAW JSON. DO NOT USE MARKDOWN BLOCKS (```json). DO NOT ADD ANY CONVERSATIONAL TEXT BEFORE OR AFTER THE JSON.
    You are an expert strict Code Reviewer evaluating generated documentation for hallucinations.
Your task is to determine if each claim made in the documentation can be STRICTLY proven by looking at the provided source code.
If a claim mentions a parameter, logic, or exception that does not exist in the code, it is a hallucination.

Claims extracted from documentation:
{claims_text}

Source Code (Ground Truth):
{input_code}

For each claim, return a verdict of 'yes' (if it is fully supported by the code) or 'no' (if it is a hallucination or contradicts the code), and provide a brief technical reason based on the source code.
Return a JSON with a 'verdicts' array, where each element contains 'claim', 'verdict' ('yes' or 'no'), and 'reason'.

"""

    @staticmethod
    def generate_reason(hallucinated_reasons: List[str], score: float) -> str:
        reasons = "\n".join(hallucinated_reasons)
        return f"""
    CRITICAL: OUTPUT ONLY VALID RAW JSON. DO NOT USE MARKDOWN BLOCKS (```json). DO NOT ADD ANY CONVERSATIONAL TEXT BEFORE OR AFTER THE JSON.
    The code faithfulness score is {score} out of 1.0.
Below are the reasons why some claims from the documentation were flagged as hallucinations or unprovable from the code:
{reasons}

Generate a brief final explanation (reason) summarizing the hallucinations found in the documentation. Emphasize what the AI falsely invented that wasn't in the original code.
Return a JSON with a single 'reason' field.
"""