from typing import List

class SemanticCompletenessTemplate:
    @staticmethod
    def extract_facts(expected_output: str) -> str:
        return f"""You are provided with a reference technical documentation (golden standard).
Your task is to extract a list of all key facts, requirements, and crucial details from it.
Break down the document into atomic, standalone facts.

Reference Documentation:
{expected_output}

Return the result in JSON format containing a 'facts' array of strings.
"""

    @staticmethod
    def evaluate_facts(actual_output: str, expected_facts: List[str]) -> str:
        facts_text = "\n".join([f"- {fact}" for fact in expected_facts])
        return f"""Evaluate whether the following expected facts are present in the generated documentation.
For each fact, output 'yes' if it is semantically covered in the generated documentation, or 'no' if it is missing, contradicted, or poorly explained. Provide a brief reason for each.

Expected Facts (Golden Standard):
{facts_text}

Generated Documentation (To be evaluated):
{actual_output}

Return a JSON with a 'verdicts' array, where each element contains 'fact', 'verdict' ('yes' or 'no'), and 'reason'.
"""

    @staticmethod
    def generate_reason(missing_facts_reasons: List[str], score: float) -> str:
        reasons = "\n".join(missing_facts_reasons)
        return f"""The semantic completeness score is {score} out of 1.0.
The generated documentation missed or misrepresented the following key facts from the reference:
{reasons}

Generate a brief final explanation (reason) for the entire metric, summarizing what important information was omitted compared to the golden standard.
Return a JSON with a single 'reason' field.
"""