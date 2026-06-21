import os
from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase

from C2D_on_functions_metrics.semantic_equivalence.semantic_equivalence_metric import SemanticEquivalenceMetric
from C2D_on_functions_metrics.code_faithfulness.code_faithfulness_metric import CodeFaithfulnessMetric
from C2D_on_functions_metrics.documentation_completeness.documentation_completeness_metric import DocumentationCompletenessMetric
from C2D_on_functions_metrics.code_coherence.code_coherence_metric import CodeCoherenceMetric

# Source code (for example, the implementation of the system logger for the kernel)
mock_input_code = """
#define LOG_BUF_SIZE 1024
char log_buf[LOG_BUF_SIZE];
int log_head = 0;

void pr_msg(const char *fmt, ...) {
    // Внутренняя реализация добавления отформатированного сообщения
    // в циклический буфер ядра (dmesg-like)
    // ...
    if (log_head >= LOG_BUF_SIZE) {
        log_head = 0; // Закольцовываем буфер
    }
}
"""

# Reference documentation (perfect description from a Senior Developer)
mock_expected_output = """
The `pr_msg` function formats and appends a diagnostic message to the kernel's internal cyclic log buffer.
It serves as the primary logging mechanism for kernel-level events and diagnostics.
Parameters:
- fmt: A format string specifying how subsequent arguments are converted for output.
- ... : Variable arguments corresponding to the format string.
Side effects: Modifies the global `log_buf` array and updates the `log_head` index.
"""

# Generated documentation (We do it specially with "jambs")
# Errors here:
# 1. Hallucination (about NetworkException - there is no network in the code).
# 2. Incompleteness (variable arguments `...` are not described).
# 3. Poor coherence (line-by-line translation of the code: "if log_head is greater...").
mock_actual_output_bad = """
`pr_msg` writes a log message. 
It takes a parameter `fmt` which is a constant character pointer.
The function writes to `log_buf`. Then it checks if `log_head` is greater than or equal to `LOG_BUF_SIZE`. If it is, it sets `log_head` to 0.
Throws `NetworkException` if the external logging server is unavailable.
"""

mock_actual_output_good = """
The `pr_msg` function writes a formatted diagnostic message into the cyclic log buffer. 
It operates as the core logging mechanism for system-level events.

Parameters:
- `fmt`: A pointer to a constant format string that determines how the following arguments are parsed and printed.
- `...`: A variable number of arguments that correspond to the format specifiers in the `fmt` string.

Side Effects:
- Modifies the global `log_buf` character array by appending the newly formatted string.
- Updates the global `log_head` index. If the buffer limit (`LOG_BUF_SIZE`) is reached, it wraps the index back to 0, ensuring continuous cyclic logging without memory overflow.
"""


semantic_metric = SemanticEquivalenceMetric(threshold=0.7, model="gpt-4.1-mini", include_reason=True)
faithfulness_metric = CodeFaithfulnessMetric(threshold=0.8, model="gpt-4.1-mini", include_reason=True)
completeness_metric = DocumentationCompletenessMetric(threshold=0.8, model="gpt-4.1-mini", include_reason=True)
coherence_metric = CodeCoherenceMetric(threshold=0.8, model="gpt-4.1-mini", include_reason=True)


test_case_bad = LLMTestCase(
    input=mock_input_code,
    expected_output=mock_expected_output,
    actual_output=mock_actual_output_bad
)

test_case_good = LLMTestCase(
    input=mock_input_code,
    expected_output=mock_expected_output,
    actual_output=mock_actual_output_good
)


metrics_to_run = [
    ("Semantic Equivalence", semantic_metric),
    ("Code Faithfulness", faithfulness_metric),
    ("Documentation Completeness", completeness_metric),
    ("Code Coherence / Developer Utility", coherence_metric)
]

print("Starting DeepEval Metrics Test...\n" + "="*40)

for metric_name, metric in metrics_to_run:
    print(f"Running {metric_name} evaluation...")

    metric.measure(test_case_bad)
    
    print(f"Status: {'PASSED' if metric.is_successful() else 'FAILED'}")
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print("-" * 40)
    
for metric_name, metric in metrics_to_run:
    print(f"Running {metric_name} evaluation...")
    
    metric.measure(test_case_good)
    
    print(f"Status: {'PASSED' if metric.is_successful() else 'FAILED'}")
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print("-" * 40)

print("All evaluations completed.")