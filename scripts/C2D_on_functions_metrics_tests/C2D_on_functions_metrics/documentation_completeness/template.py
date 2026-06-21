from typing import List
from .schema import RequiredDocElement

class DocumentationCompletenessTemplate:
    @staticmethod
    def extract_requirements(input_code: str) -> str:
        return f"""You are an expert Code Analyzer.
Analyze the provided source code and generate a strict checklist of elements that MUST be covered in its documentation for it to be considered complete.

Rules for the checklist:
1. ALWAYS include 'General Purpose' (what the function does).
2. Include EVERY parameter defined in the function signature.
3. Include 'Return Value' if the function returns anything other than None/void.
4. Include specific exceptions if the function explicitly raises or throws them.

Source Code:
{input_code}

Return the result in JSON format, containing an 'elements' array where each item has 'element_name' and 'justification'.
CRITICAL: OUTPUT ONLY VALID RAW JSON. DO NOT USE MARKDOWN BLOCKS (```json). DO NOT ADD ANY CONVERSATIONAL TEXT BEFORE OR AFTER THE JSON.
"""

    @staticmethod
    def generate_verdicts(elements: List[RequiredDocElement], actual_output: str) -> str:
        checklist_text = "\n".join([f"- {e.element_name} ({e.justification})" for e in elements])
        return f"""You are a Documentation Reviewer.
Your task is to check if the generated documentation successfully covers all the required elements from the provided checklist.

Checklist of required elements:
{checklist_text}

Generated Documentation (Actual Output):
{actual_output}

For each element on the checklist, return a verdict of 'yes' (if it is clearly explained in the documentation) or 'no' (if it is completely missing, skipped, or too vaguely described).
Return a JSON with a 'verdicts' array, where each item contains 'element_name', 'verdict' ('yes' or 'no'), and 'reason'.
CRITICAL: OUTPUT ONLY VALID RAW JSON. DO NOT USE MARKDOWN BLOCKS (```json). DO NOT ADD ANY CONVERSATIONAL TEXT BEFORE OR AFTER THE JSON.
"""

    @staticmethod
    def generate_reason(missing_elements: List[str], score: float) -> str:
        reasons = "\n".join(missing_elements)
        return f"""The documentation completeness score is {score} out of 1.0.
Below are the elements that were missing or poorly explained in the documentation:
{reasons}

Generate a brief final explanation summarizing why the documentation is incomplete. Focus on the most critical omissions (e.g., missing parameters or return values). If the score is 1.0, just state that it is fully complete.
Return a JSON with a single 'reason' field.
CRITICAL: OUTPUT ONLY VALID RAW JSON. DO NOT USE MARKDOWN BLOCKS (```json). DO NOT ADD ANY CONVERSATIONAL TEXT BEFORE OR AFTER THE JSON.
"""