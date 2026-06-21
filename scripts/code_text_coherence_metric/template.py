from typing import List
from schema import CodeExplanationPair

class CodeCoherenceTemplate:
    @staticmethod
    def extract_pairs(actual_output: str) -> str:
        return f"""You are provided with generated technical documentation in Markdown format.
Your task is to extract all code blocks and their corresponding explanatory text.
If a code block has no explanation, return an empty string in the 'explanation' field.

Documentation:
{actual_output}

Return the result in JSON format, containing a 'pairs' array with 'code' and 'explanation' fields.
"""

    @staticmethod
    def generate_verdicts(pairs: List[CodeExplanationPair]) -> str:
        pairs_text = "\n\n".join([f"Code:\n{p.code}\nExplanation: {p.explanation}" for p in pairs])
        return f"""Evaluate how well the explanatory text describes the corresponding code block.
For each pair, provide a verdict of 'yes' (if the explanation is accurate, coherent, and relates to the code) or 'no' (if the explanation is missing, contains errors, or is unrelated to the code), and also provide a reason.

Pairs to evaluate:
{pairs_text}

Return a JSON with a 'verdicts' array, where each element contains 'code', 'verdict' ('yes' or 'no'), and 'reason'.
"""

    @staticmethod
    def generate_reason(irrelevant_statements: List[str], score: float) -> str:
        reasons = "\n".join(irrelevant_statements)
        return f"""The code-text coherence score is {score} out of 1.0.
Below are the reasons why some code blocks were deemed poorly explained:
{reasons}

Generate a brief final explanation (reason) for the entire metric, summarizing why the documentation received this score.
Return a JSON with a single 'reason' field.
"""