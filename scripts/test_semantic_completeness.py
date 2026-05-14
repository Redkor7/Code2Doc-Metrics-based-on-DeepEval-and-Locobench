from deepeval.test_case import LLMTestCase
from semantic_completeness.semantic_completeness import SemanticCompletenessMetric 
from giga_chat import CustomGigaChat
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GigaChat_API_Key")

# 1. Prepare the reference data (Golden Standard)
# This is the ideal response that contains all key facts.
mock_expected_output = """
# Reference Setup Instructions
1. The project strictly requires Python version 3.10 or higher.
2. Before running, you must install the dependencies: `pip install -r requirements.txt`.
3. The PostgreSQL database must be running on the standard port 5432.
4. The local server is started with the command `python main.py`.
5. By default, the server is available at http://localhost:8000.
"""

# 2. Prepare the actual data (Output from the tested LLM)
# Here we intentionally "forgot" to mention the Python version, package installation, and DB port.
mock_actual_output = """
# Project Setup Instructions

To start the project, you will need a running PostgreSQL database.

Start the local server using the following script:
```bash
python main.py
After that, the application will be available in the browser at http://localhost:8000.
Ensure that the database is working correctly.
""" 

# 3. Initialize GigaChat
# Replace "api" with your actual key or leave as is if the key is pulled from .env
gigachat_llm = CustomGigaChat(credentials=API_KEY)

# 4. Create the metric
metric = SemanticCompletenessMetric(
threshold=0.8,
model=gigachat_llm,
include_reason=True,
verbose_mode=True # Enabled to see the process of fact extraction and judging
)

# 5. Create the test case
test_case = LLMTestCase(
input="Write instructions for starting the local server and the database.",
actual_output=mock_actual_output,
expected_output=mock_expected_output
)

# 6. Run the measurement
print("")
print("Running Semantic Completeness evaluation...")
print("\n")
metric.measure(test_case)

# 7. Print the results
print("\n======================================================================")
print(f"Successful: {metric.is_successful()}")
print(f"Final score: {metric.score}")
print(f"Reason:\n{metric.reason}")
print("======================================================================")