from deepeval.test_case import LLMTestCase
from code_text_coherence import CodeCoherenceMetric 
from giga_chat import CustomGigaChat

# 1. Prepare the test data (output data from the tested LLM)
# Here we intentionally make the first part coherent and the second part meaningless, 
# so that the metric returns a score below 1.0 and explains the reason.
mock_llm_output = """
# Project Setup Instructions

First, you need to start the local server. Use the following script:

```python
import http.server
import socketserver

PORT = 8080
Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("Server started on port", PORT)
    httpd.serve_forever()
````

This code imports built-in Python modules and starts a simple HTTP server on port 8080. It will serve files from the directory where the script was launched.

Next, let's create the main page:

HTML

```
<!DOCTYPE html>
<html>
<body>
    <h1>Welcome</h1>
</body>
</html>
```

In order to connect to the PostgreSQL database, you need to use the psycopg2 library. Make sure to check your firewall settings so that port 5432 is open. """

# 2. Initialize GigaChat
gigachat_llm = CustomGigaChat(credentials="api")

# 3. Create a metric with a custom LLM as a judge
metric = CodeCoherenceMetric( threshold=0.5, model=gigachat_llm, include_reason=True, verbose_mode=True )

# 4. Create a test case
test_case = LLMTestCase( input="Write instructions for starting a local server and creating a starter HTML page.", actual_output=mock_llm_output )

# 5. Measure Coherence
print("Running Code Coherence evaluation...") 
metric.measure(test_case)

# 6. Print results
print(f"Successful: {metric.is_successful()}") 
print(f"Final score: {metric.score}") 
print(f"Reason: {metric.reason}")