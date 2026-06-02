from typing import List
from .schema import SemanticClaim

class SemanticEquivalenceTemplate:
    @staticmethod
    def extract_claims(expected_output: str) -> str:
        return f"""You are provided with a reference (expected) technical documentation.
Your task is to break down this documentation into a list of atomic, factual claims. 
Each claim should represent a single piece of information about the function's behavior, parameters, or return values.

Reference Documentation:
{expected_output}

Return the result in JSON format, containing a 'claims' array where each element has a 'claim' string field.
CRITICAL: OUTPUT ONLY VALID RAW JSON. DO NOT USE MARKDOWN BLOCKS (```json). DO NOT ADD ANY CONVERSATIONAL TEXT BEFORE OR AFTER THE JSON.
"""

    @staticmethod
    def generate_verdicts(claims: List[SemanticClaim], actual_output: str) -> str:
        claims_text = "\n".join([f"- {c.claim}" for c in claims])
        return f"""You are evaluating generated code documentation for semantic equivalence against a set of expected claims.
Your task is to determine if each expected claim is supported by the generated documentation.
Ignore exact word matches; focus entirely on meaning, synonyms, and technical equivalence.

Expected Claims:
{claims_text}

Generated Documentation (Actual Output):
{actual_output}

For each claim, return a verdict of 'yes' (if the meaning is present in the generated documentation) or 'no' (if it is missing or contradicted), and provide a brief reason.
Return a JSON with a 'verdicts' array, where each element contains 'claim', 'verdict' ('yes' or 'no'), and 'reason'.
CRITICAL: OUTPUT ONLY VALID RAW JSON. DO NOT USE MARKDOWN BLOCKS (```json). DO NOT ADD ANY CONVERSATIONAL TEXT BEFORE OR AFTER THE JSON.
"""

    @staticmethod
    def generate_reason(missing_claims_reasons: List[str], score: float) -> str:
        reasons = "\n".join(missing_claims_reasons)
        return f"""The semantic equivalence score is {score} out of 1.0.
Below are the reasons why some expected claims were NOT found in the generated documentation:
{reasons}

Generate a brief final explanation (reason) summarizing why the generated documentation received this score. Focus on what core semantic meaning was lost or altered compared to the reference.
Return a JSON with a single 'reason' field.
CRITICAL: OUTPUT ONLY VALID RAW JSON. DO NOT USE MARKDOWN BLOCKS (```json). DO NOT ADD ANY CONVERSATIONAL TEXT BEFORE OR AFTER THE JSON.
"""