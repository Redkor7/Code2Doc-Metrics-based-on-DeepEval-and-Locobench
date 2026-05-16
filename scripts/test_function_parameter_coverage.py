from deepeval.test_case import LLMTestCase
from function_parameter_coverage.function_parameter_coverage import FunctionParameterCoverageMetric 
from giga_chat import CustomGigaChat
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GigaChat_API_Key")

# 1. Prepare the test data
# Первая функция описана полностью (параметры name, uppercase).
# Вторая функция описана частично (host описан, port и retry_count - нет).
mock_llm_output = """
# API Client Documentation

Below are the core functions to interact with our system.

### Greeting Function
```python
def generate_greeting(name: str, uppercase: bool = False) -> str:
    pass
This function generates a welcome message for the user.
Parameters:

name: A string representing the username. We need this to personalize the message.

uppercase: A boolean flag. Set this to True if you need the entire output to be in capital letters for emphasis.

Database Connection
Python
def connect_to_db(host: str, port: int, retry_count: int = 3):
    pass
Use this function to establish a connection to the PostgreSQL backend.
You must provide the host parameter, which is the IP address of the database server needed to route the traffic.
"""

gigachat_llm = CustomGigaChat(credentials=API_KEY)

metric = FunctionParameterCoverageMetric(
threshold=0.8,
model=gigachat_llm,
include_reason=True,
verbose_mode=True
)

test_case = LLMTestCase(
input="Write technical documentation for generate_greeting and connect_to_db functions.",
actual_output=mock_llm_output
)

print("Running Function Parameter Coverage evaluation...")
metric.measure(test_case)
print(f"Successful: {metric.is_successful()}")
print(f"Final score: {metric.score}")
print(f"Reason: {metric.reason}")