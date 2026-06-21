from typing import List
from schema import FunctionExtraction

class FunctionCoverageTemplate:
    @staticmethod
    def extract_functions(actual_output: str) -> str:
        return f"""You are provided with generated technical documentation in Markdown format.
Your task is to extract all function/method signatures, their parameter names, and the corresponding explanatory text for the parameters.
If a function has no explanation, return an empty string in the 'explanation' field.

IMPORTANT JSON FORMATTING RULES:
1. 'parameters_list' MUST be a simple list of strings (just the parameter names). Do NOT use dictionaries or objects inside this list.
2. 'explanation' MUST be a single string containing the entire block of text that describes the function and its parameters.

Documentation:
{actual_output}

Return the result in JSON format, containing a 'functions' array with 'signature', 'parameters_list', and 'explanation' fields.
"""

    @staticmethod
    def generate_verdicts(functions: List[FunctionExtraction]) -> str:
        functions_text = "\n\n".join([f"Signature:\n{f.signature}\nParameters: {', '.join(f.parameters_list)}\nExplanation: {f.explanation}" for f in functions])
        return f"""Evaluate how well the explanatory text describes all the parameters of the corresponding function.
For each function, check if the explanation clearly states what EVERY parameter in the 'Parameters' list is and WHY it is needed.
Provide a verdict of 'yes' (if ALL parameters are explained adequately) or 'no' (if any parameter explanation is missing, vague, or missing the 'why'), and also provide a detailed reason.

Functions to evaluate:
{functions_text}

Return a JSON with a 'verdicts' array, where each element contains 'signature', 'verdict' ('yes' or 'no'), and 'reason'.
"""

    @staticmethod
    def generate_reason(irrelevant_statements: List[str], score: float) -> str:
        reasons = "\n".join(irrelevant_statements)
        return f"""The function parameter coverage score is {score} out of 1.0.
Below are the reasons why some functions were deemed to have poorly documented parameters:
{reasons}

Generate a brief final explanation (reason) for the entire metric, summarizing why the documentation received this score. Focus on what parameters or details were generally missing.
Return a JSON with a single 'reason' field.
"""