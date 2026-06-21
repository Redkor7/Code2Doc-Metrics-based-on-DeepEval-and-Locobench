from typing import List

class CodeCoherenceTemplate:
    @staticmethod
    def generate_verdicts(input_code: str, actual_output: str) -> str:
        return f"""You are a strict Senior Software Engineer conducting a documentation review.
Evaluate the generated documentation for the provided source code based on its coherence and actual utility to another developer.

You must evaluate exactly these three aspects:
1. 'Intent': Does the documentation explain the "WHY"? (For example, for a C function in an ext2 filesystem or an xv6 kernel, it should explain the system-level intent, like 'Extracts directory entries from the disk image', rather than just stating 'Opens a file').
2. 'Abstraction': Does the documentation avoid trivial, line-by-line translation? (It should NOT say 'increments i by 1 and loops' for a simple loop).
3. 'Clarity': Is the text written in clear, concise, and standard technical English (B1-B2 level or higher), without convoluted grammar, repetitive phrasing, or robotic GPT-style templates?

Source Code:
{input_code}

Generated Documentation (Actual Output):
{actual_output}

Return a JSON with a 'verdicts' array. Provide exactly three items, one for each aspect. 
Each item MUST have an 'aspect' field (strictly 'Intent', 'Abstraction', or 'Clarity'), a 'verdict' field ('yes' or 'no'), and a detailed 'reason' field.
CRITICAL: OUTPUT ONLY VALID RAW JSON. DO NOT USE MARKDOWN BLOCKS (```json). DO NOT ADD ANY CONVERSATIONAL TEXT BEFORE OR AFTER THE JSON.
"""

    @staticmethod
    def generate_reason(failed_aspects_reasons: List[str], score: float) -> str:
        reasons = "\n".join(failed_aspects_reasons)
        return f"""The code coherence and utility score is {score} out of 1.0.
Below are the reasons why the documentation failed certain quality aspects:
{reasons}

Generate a brief final explanation summarizing why this documentation is considered unhelpful, incoherent, or poorly written for a developer.
Return a JSON with a single 'reason' field.
CRITICAL: OUTPUT ONLY VALID RAW JSON. DO NOT USE MARKDOWN BLOCKS (```json). DO NOT ADD ANY CONVERSATIONAL TEXT BEFORE OR AFTER THE JSON.
"""