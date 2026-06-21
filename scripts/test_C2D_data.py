import os
import re
from deepeval.test_case import LLMTestCase

from C2D_on_functions_metrics.semantic_equivalence.semantic_equivalence_metric import SemanticEquivalenceMetric
from C2D_on_functions_metrics.code_faithfulness.code_faithfulness_metric import CodeFaithfulnessMetric
from C2D_on_functions_metrics.documentation_completeness.documentation_completeness_metric import DocumentationCompletenessMetric
from C2D_on_functions_metrics.code_coherence.code_coherence_metric import CodeCoherenceMetric

def parse_code_file(filepath: str) -> dict:
    """Extracts the source code from a Markdown file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = r"### Функция ID:\s*(\d+).*?```[a-zA-Z]*\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    return {int(idx): code.strip() for idx, code in matches}

def parse_markdown_text_blocks(filepath: str) -> dict:
    """Extracts text (reference and generated documentation) from a Markdown file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    parts = re.split(r"### Функция ID:\s*(\d+)", content)
    docs_dict = {}
    
    for i in range(1, len(parts), 2):
        idx = int(parts[i])
        text = parts[i+1].strip()
        text = re.sub(r"\n---\s*$", "", text).strip()
        docs_dict[idx] = text
        
    return docs_dict

def main():
    CODE_FILE_PATH = "functions_for_llm_400.md"
    REFERENCE_DOCS_PATH = "reference_docs_400.md"
    GENERATED_DOCS_PATH = "doc_from_deepseek_400.md"
    RESULTS_TABLE_PATH = "models_comparison.md"     

    print("Parsing files...")
    code_dict = parse_code_file(CODE_FILE_PATH)
    reference_dict = parse_markdown_text_blocks(REFERENCE_DOCS_PATH)
    generated_dict = parse_markdown_text_blocks(GENERATED_DOCS_PATH)

    # Initializing metrics
    semantic_metric = SemanticEquivalenceMetric(threshold=0.7, model="gpt-4.1-mini", include_reason=True)
    faithfulness_metric = CodeFaithfulnessMetric(threshold=0.8, model="gpt-4.1-mini", include_reason=True)
    completeness_metric = DocumentationCompletenessMetric(threshold=0.8, model="gpt-4.1-mini", include_reason=True)
    coherence_metric = CodeCoherenceMetric(threshold=0.8, model="gpt-4.1-mini", include_reason=True)

    metrics_to_run = [
        ("Semantic Equivalence", semantic_metric),
        ("Code Faithfulness", faithfulness_metric),
        ("Documentation Completeness", completeness_metric),
        ("Code Coherence / Developer Utility", coherence_metric)
    ]
    
    # Dictionary for storing lists of scores for each metric
    scores_accumulator = {name: [] for name, _ in metrics_to_run}

    print("Starting DeepEval testing...\n" + "="*40)

    # Looking for common IDs in all three dictionaries
    common_ids = sorted(set(code_dict.keys()) & set(reference_dict.keys()) & set(generated_dict.keys()))
    
    if not common_ids:
        print("Error: No matching IDs found between files.")
        return

    for func_id in common_ids:
        print(f"\nEvaluating Function ID: {func_id}")
        
        test_case = LLMTestCase(
            input=code_dict[func_id],
            actual_output=generated_dict[func_id],
            expected_output=reference_dict[func_id]
        )

        for metric_name, metric in metrics_to_run:
            try:
                metric.measure(test_case)
                scores_accumulator[metric_name].append(metric.score) 
                print(f"[{metric_name}] Score: {metric.score}")
            except Exception as e:
                print(f"[{metric_name}] ERROR: {str(e)}")

    print("\n" + "="*40)
    print("Calculating average values and writing to table...")

    averages = {}
    for name in scores_accumulator:
        scores = scores_accumulator[name]
        if scores:
            averages[name] = sum(scores) / len(scores)
        else:
            averages[name] = 0.0

    # Writing to a Markdown file
    file_exists = os.path.isfile(RESULTS_TABLE_PATH)
    
    with open(RESULTS_TABLE_PATH, "a", encoding="utf-8") as f:
        # If there is no file, first draw the header of the table
        if not file_exists:
            headers = ["Model Name"] + [name for name, _ in metrics_to_run]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        
        # Forming a string with the results (rounded to 3 decimal places)
        row_values = ["deepSeek-V3"] + [f"{averages[name]:.3f}" for name, _ in metrics_to_run]
        f.write("| " + " | ".join(row_values) + " |\n")

    print(f"Done! Results for deepSeek-V3 saved to {RESULTS_TABLE_PATH}")

if __name__ == "__main__":
    main()