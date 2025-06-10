# autoeval-to-mongodb

![](braintrust-autoevals.png)

![](https://private-user-images.githubusercontent.com/114057324/324758306-3a4b863e-cea3-4d89-a6f5-24a4ee44cfd4.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDk1MzQ2NzMsIm5iZiI6MTc0OTUzNDM3MywicGF0aCI6Ii8xMTQwNTczMjQvMzI0NzU4MzA2LTNhNGI4NjNlLWNlYTMtNGQ4OS1hNmY1LTI0YTRlZTQ0Y2ZkNC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNjEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDYxMFQwNTQ2MTNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1iNzRhNjI0ZjAyNDk4ZTIzOWJhZTIwYmI1ODA2YzljNTFmZWY1MzEzMGJiODdhYjUzZGI5OTVjNTNlNjE2NmY0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.1eWxrVw3qJ-31rRMw-662yHdRZ0lKKViXKiddkw7haM)

---

### **autoeval-to-mongodb: An End-to-End Blueprint for Building and Validating a RAG System**

This project provides a complete, runnable blueprint for building, testing, and evaluating a Retrieval-Augmented Generation (RAG) application. In today's AI landscape, creating a RAG system is only half the battle; ensuring its reliability and accuracy is paramount. This repository addresses that challenge directly by integrating automated evaluation into the core development workflow.

We demonstrate a practical RAG use case: a movie lookup system that finds film titles based on plot descriptions. The project moves beyond theory and provides a hands-on implementation using a modern, powerful tech stack.

**Key Components:**

* **Database:** A local **MongoDB Atlas** instance running in **Docker**, populated with a sample movie dataset.
* **Intelligence Engine:** **MongoDB Atlas Vector Search** for semantic retrieval, coupled with **Azure OpenAI** for generating embeddings and final responses.
* **Core Logic:** A **Python** script that orchestrates the entire RAG pipeline from user prompt to final answer.
* **Automated Validation:** The **`autoevals` library** is used to systematically measure the system's factuality against a predefined test dataset, producing a quantitative performance score.

The outcome is a tangible demonstration of how to not only build a sophisticated AI system but also to generate a data-driven report card on its performance. This allows developers to iterate, measure improvements, and build more trustworthy AI applications.

### Prerequisites

* **MongoDB Tools:**
  * **mongosh:** The official MongoDB shell for interacting with MongoDB databases.
  * **mongorestore:** A tool for restoring data from a dump file to a MongoDB database.
* **Docker:** Installed on your system ([https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/))
* **wget or curl:** Installed on your system (package managers usually handle this)


### Setting Up a Local Atlas Environment

1. **Pull the Docker Image:**

   * **Latest Version:**
     ```bash
     docker pull mongodb/mongodb-atlas-local
     ```

2. **Run the Database:**

   ```bash
   docker run -p 27017:27017 mongodb/mongodb-atlas-local
   ```
   This command runs the Docker image, exposing port 27017 on your machine for connecting to the database.

### Using Sample Datasets with MongoDB

This section demonstrates downloading and exploring a sample dataset for MongoDB on your local system.

#### Downloading the Dataset

There's a complete sample dataset available for MongoDB. Download it using either `wget` or `curl`:

* **Using wget:**

```bash
wget https://atlas-education.s3.amazonaws.com/sampledata.archive
```

* **Using curl:**

```bash
curl https://atlas-education.s3.amazonaws.com/sampledata.archive -o sampledata.archive
```

**Note:**

* Ensure you have `wget` or `curl` installed.
* The downloaded file will be named `sampledata.archive`.

#### Restoring the Dataset

Before restoring, ensure you have a local `mongod` instance running (either existing or newly started). This instance will host the dataset.

**To restore the dataset:**

```bash
mongorestore --archive=sampledata.archive
```

This command uses the `mongorestore` tool to unpack the downloaded archive (`sampledata.archive`) and populate your local `mongod` instance with the sample data.

### Creating an Atlas Vector Search Index with mongosh

**Steps:**

1. **Connect to Local Atlas Cluster:**

   Use `mongosh` to connect to the database:

   ```bash
   mongosh "mongodb://localhost/?directConnection=true"
   ```

2. **Switch to the Database:**

   Select the database containing the collection you want to index:

   ```javascript
   use sample_mflix
   ```

3. **Create the Index:**

   ```javascript
   db.embedded_movies.createSearchIndex(
       "embeddings_1_search_index",
       "vectorSearch", // index type
       {
           fields: [
               {
                   "type": "vector",
                   "numDimensions": 1536,
                   "path": "plot_embedding",
                   "similarity": "cosine"
               },
               {"type":"filter","path":"genres"},
               {"type":"filter","path":"type"}
           ]
       }
   );
   ```

4. **Check the Index Status:**

   ```javascript
   db.embedded_movies.getSearchIndexes()
   ```

5. **Wait for Status 'READY'**:

   A successful response will look similar to this:

   ```json
   [
       {
           "id": "...",
           "name": "embeddings_1_search_index",
           "type": "vectorSearch",
           "status": "READY",
           "queryable": true,
           "latestVersion": 0,
           "latestDefinition": {
               "fields": [
                   {
                       "type": "vector",
                       "numDimensions": 1536,
                       "path": "plot_embedding",
                       "similarity": "cosine"
                   }
               ]
           }
       }
   ]
   ```

   -----


   # FULL CODE

```python
import os  
import sys  
from pymongo import MongoClient  
from autoevals import Factuality, init  
"""  
Factuality Check for Azure OpenAI RAG Task  
comparing a submitted answer to an expert answer on a given question  
"""  
from openai import AzureOpenAI, APIError, AuthenticationError  
import json  
from datetime import datetime  
  
# --- Configuration ---  
# It's best practice to set secrets as environment variables.  
# load the environment variables from a .env file if it exists  
from dotenv import load_dotenv  
load_dotenv()  
# Ensure the environment variables are set correctly.  
  
# 2. Azure OpenAI Configuration  
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")  
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")  
AZURE_API_VERSION = "2023-12-01-preview"  # Update to your API version  
  
# Deployment names must match those in your Azure AI Studio.  
AZURE_CHAT_DEPLOYMENT_NAME = "gpt-4o"  # Replace with your GPT-4 deployment name  
# Add a deployment name for the model used to create embeddings for vector search.  
AZURE_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"  # Replace with your embedding model deployment  
  
# 3. MongoDB Configuration (Inspired by the provided Flask App)  
# In your terminal: `export MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority"`  
MONGO_URI = "" # os.environ.get("MONGO_URI")  # This is required for the script to run  

DB_NAME = "sample_mflix"  # e.g., "knowledge_base"  
COLLECTION_NAME = "embedded_movies"  # e.g., "embeddings"  
VECTOR_INDEX_NAME = "embeddings_1_search_index"  # e.g., "vector_index"  
  
# --- Client Initialization ---  
# Initialize Azure OpenAI Client  
azure_client = None  
try:  
    if not AZURE_ENDPOINT or not AZURE_API_KEY:  
        raise ValueError("Azure endpoint or API key is not configured. Please set the environment variables.")  
  
    azure_client = AzureOpenAI(  
        api_version=AZURE_API_VERSION,  
        azure_endpoint=AZURE_ENDPOINT,  
        api_key=AZURE_API_KEY,  
    )  
    # Initialize autoevals with the client  
    init(azure_client)  
except Exception as e:  
    print(f"Fatal Error: Could not initialize AzureOpenAI client. Check your configuration. Details: {e}", file=sys.stderr)  
    sys.exit(1)  
  
# Initialize MongoDB Client  
mongo_client = None  
if MONGO_URI:  
    try:  
        mongo_client = MongoClient(MONGO_URI)  
        db = mongo_client[DB_NAME]  
        collection = db[COLLECTION_NAME]  
        print("Successfully connected to MongoDB.")  
    except Exception as e:  
        print(f"Fatal Error: Could not connect to MongoDB. Check MONGO_URI. Details: {e}", file=sys.stderr)  
        sys.exit(1)  
else:  
    print("Warning: MONGO_URI is not set. The RAG task will not be able to query the database.", file=sys.stderr)  
  
  
# --- Helper Functions for RAG ---  
  
def get_embedding(text: str, model: str = AZURE_EMBEDDING_DEPLOYMENT_NAME) -> list:  
    """Generates a vector embedding for a given text using Azure OpenAI."""  
    if not azure_client:  
        raise ValueError("Azure client is not initialized.")  
    try:  
        response = azure_client.embeddings.create(input=[text], model=model)  
        return response.data[0].embedding  
    except Exception as e:  
        print(f"Error generating embedding: {e}", file=sys.stderr)  
        return []  
  
def perform_vector_search(vector: list) -> list:  
    """  
    Performs a $vectorSearch query in MongoDB to find relevant documents.  
    This function is similar to the `knowledge_repo.vector_search` in the provided Flask app.  
    """  
    if not mongo_client:  
        print("Cannot perform vector search, MongoDB client not initialized.", file=sys.stderr)  
        return []  
    
    # This is the aggregation pipeline that uses the vector index  
    pipeline = [  
        {  
            "$vectorSearch": {  
                "index": VECTOR_INDEX_NAME,  
                "path": "plot_embedding",  # The field in your documents that contains the vector  
                "queryVector": vector,  
                "numCandidates": 200,  # Number of candidates to consider  
                "limit": 5             # Number of results to return  
            }  
        },  
        {  
            "$project": {  
                "_id": 0,  
                "score": {"$meta": "vectorSearchScore"},  
                "title": 1,  
                "plot": 1,
                "year": 1,  
            }  
        }  
    ]  
    try:  
        results = list(collection.aggregate(pipeline))  
        return results  
    except Exception as e:  
        print(f"Error during vector search in MongoDB: {e}", file=sys.stderr)  
        return []  
  
  
# --- Task Definition (RAG Workflow) ---  
  
def run_rag_task(input_prompt: str) -> str:  
    """  
    Executes the full Retrieval-Augmented Generation (RAG) task:  
    1. Generates an embedding for the input.  
    2. Retrieves context from MongoDB via vector search.  
    3. Generates a final response using the retrieved context.  
    """  
    if not azure_client:  
        return "Error: Azure OpenAI client is not initialized."  
    if not mongo_client:  
        return "Error: MongoDB client is not initialized."  
  
    # 1. Get query vector  
    print(f"Generating embedding for query: '{input_prompt}'")  
    query_vector = get_embedding(input_prompt)  
    if not query_vector:  
        return "Error: Failed to generate embedding for the query."  
  
    # 2. Query MongoDB for context  
    print("Performing vector search in MongoDB...")  
    context_docs = perform_vector_search(query_vector)  
    if not context_docs:  
        print("No context found from vector search.", file=sys.stderr)  
        # Fallback: answer without context  
        context_str = "No specific context was found."  
    else:  
        print(f"Retrieved {len(context_docs)} documents from MongoDB.")  
        # Format the context for the prompt  
        context_str = "\n".join([f"- {doc['title']} ({doc['year']}) \n\n {doc['plot']}" for doc in context_docs])  
  
    # 3. Generate response from the chat model with the retrieved context  
      
    try:  
        response = azure_client.chat.completions.create(  
            model=AZURE_CHAT_DEPLOYMENT_NAME,  
            messages=[  
                    {  
                        "role": "system",  
                        "content": f"""  
Provide the answer in JSON, with the answer for the user in the field `response` and whether the provided context was used in `used_provided_context`.  
Below in the context, you will find bits of the plot of movies.
If the context is relevant to the question, use it to provide a more accurate answer.
[context]  
{context_str}  
[/context]  
                        """  
                    },  
                    {  
                        "role": "user",  
                        "content": f"""  
[response_criteria]  
- Provide a concise answer to the question based on the context and what you know.  
- Respond ONLY with the title of the movie that best matches the question.
[/response_criteria]  
  
[question]  
{input_prompt}  
[/question]  
                        """  
                    }  
                ],  
            stream=False,  
            response_format={"type": "json_object"}  
        )  
        json_answer = json.loads(response.choices[0].message.content.strip())  
        return json_answer["response"].strip() if "response" in json_answer else "No response provided."  
    except Exception as e:  
        print(f"An unexpected error occurred during chat completion: {e}", file=sys.stderr)  
        return ""  
  
# Define a shared dataset for local validation.
TEST_DATASET = [
  {
    "id": 1,
    "input": "Movie where a cannibal genius helps an FBI trainee hunt another killer.",
    "expected": "The Silence of the Lambs"
  },
  {
    "id": 2,
    "input": "A kid in a time-traveling DeLorean has to make his parents fall in love at a high school dance.",
    "expected": "Back to the Future"
  },
  {
    "id": 3,
    "input": "Giant shark terrorizes a beach town and the police chief says they're 'gonna need a bigger boat'.",
    "expected": "Jaws"
  },
  {
    "id": 4,
    "input": "A hacker has to choose between a red pill and a blue pill to see the true nature of reality.",
    "expected": "The Matrix"
  },
  {
    "id": 5,
    "input": "An old woman tells her story of a tragic romance on a doomed ship and a lost blue diamond necklace.",
    "expected": "Titanic"
  }
]
  
def test_rag_task_with_factuality():  
    """Runs the RAG task on test data, evaluates factuality, and stores results in a JSON object."""  
    factual = Factuality()  
  
    test_run = {  
        "timestamp": datetime.utcnow().isoformat(),  
        "test_cases": [],  
        "average_score": None  
    }  
    total_score = 0  
  
    for idx, test_case in enumerate(TEST_DATASET):  
        input_prompt = test_case["input"]  
        expected_output = test_case["expected"].strip()  
  
        generated_output = run_rag_task(input_prompt)  
  
        result = factual.eval(  
            input=input_prompt,  
            output=generated_output,  
            expected=expected_output  
        )  
  
        test_case_result = {  
            "test_case_id": test_case.get("id", idx + 1),  
            "input_prompt": input_prompt,  
            "expected_output": expected_output,  
            "generated_output": generated_output,  
            "factuality_score": result.score,  
        }  
  
        test_run["test_cases"].append(test_case_result)  
        total_score += result.score  
  
        print(f"\n{'='*10} Test Case {idx+1} {'='*10}")  
        print(f"Input Prompt:\n{input_prompt}")  
        print(f"Expected Output:\n{expected_output}")  
        print(f"Generated Output:\n{generated_output}")  
        print(f"Factuality Score: {result.score}")  
        print(f"{'='*30}")  
  
    # Calculate average score  
    average_score = total_score / len(TEST_DATASET) if TEST_DATASET else 0  
    test_run["average_score"] = average_score  
  
    print("\n--- Test Run Summary ---")  
    print(f"Average Factuality Score: {average_score}")  
  
    # Pretty-print the JSON object  
    print("\n--- JSON Object for Test Run ---")  
    print(json.dumps(test_run, indent=4))  
  
    # Optionally, return the test_run object for further processing or storage  
    return test_run  
  
def validate_locally():  
    """  
    Runs the RAG task on a local dataset and prints the 'expected' vs 'got'  
    output directly to the console for immediate validation.  
    """  
    print("--- Starting Local Validation ---")  
  
    for i, item in enumerate(TEST_DATASET):  
        print(f"\n{'='*10} Test Case {i+1} {'='*10}")  
        input_prompt = item["input"]  
        expected_output = item["expected"].strip()  
  
        got_output = run_rag_task(input_prompt)  
  
        print("\n--- Comparison ---")  
        print(f"INPUT:\n{input_prompt}")  
        print(f"EXPECTED:\n{expected_output}")  
        print(f"GOT:\n{got_output}")  
        print(f"{'='*33}")  
  
        # Simple check  
        if expected_output.lower() in got_output.lower():  
             print("RESULT:   Contains expected text.")  
        else:  
             print("RESULT:   Does NOT contain expected text.")  
  
if __name__ == "__main__":  
    # Run the test with factuality scoring and produce a structured JSON object  
    test_run_data = test_rag_task_with_factuality()  


"""
==============================

--- Test Run Summary ---
Average Factuality Score: 0.64

--- JSON Object for Test Run ---
{
    "timestamp": "2025-06-10T05:39:13.523631",
    "test_cases": [
        {
            "test_case_id": 1,
            "input_prompt": "Movie where a cannibal genius helps an FBI trainee hunt another killer.",
            "expected_output": "The Silence of the Lambs",
            "generated_output": "The Silence of the Lambs",
            "factuality_score": 1
        },
        {
            "test_case_id": 2,
            "input_prompt": "A kid in a time-traveling DeLorean has to make his parents fall in love at a high school dance.",
            "expected_output": "Back to the Future",
            "generated_output": "Back to the Future Part II",
            "factuality_score": 0
        },
        {
            "test_case_id": 3,
            "input_prompt": "Giant shark terrorizes a beach town and the police chief says they're 'gonna need a bigger boat'.",
            "expected_output": "Jaws",
            "generated_output": "Jaws (1975)",
            "factuality_score": 0.6
        },
        {
            "test_case_id": 4,
            "input_prompt": "A hacker has to choose between a red pill and a blue pill to see the true nature of reality.",
            "expected_output": "The Matrix",
            "generated_output": "The Matrix",
            "factuality_score": 1
        },
        {
            "test_case_id": 5,
            "input_prompt": "An old woman tells her story of a tragic romance on a doomed ship and a lost blue diamond necklace.",
            "expected_output": "Titanic",
            "generated_output": "Titanic (1996)",
            "factuality_score": 0.6
        }
    ],
    "average_score": 0.64
}
"""
```

## APPENDIX

### **Improving the Score with Simple Prompt Engineering**

Our current average factuality score is **0.64**. While a good start, we can easily improve it by addressing the partial-credit scores. The test cases for "Jaws" and "Titanic" received a `0.6` score because the model returned the title *and* the year (e.g., "Jaws (1975)"), which doesn't exactly match the `expected` output of just the title.

This is a classic opportunity for prompt engineering. The model is being helpful, but *too* helpful. We can make our instructions more specific to get the precise output format we need.

**Original `response_criteria` in the prompt:**
```
[response_criteria]
- Provide a concise answer to the question based on the context and what you know.
- Respond ONLY with the title of the movie that best matches the question.
[/response_criteria]
```

**Improved `response_criteria`:**
By adding one explicit negative constraint, we can guide the model to the perfect answer.

```
[response_criteria]
- Provide a concise answer to the question based on the context and what you know.
- Respond ONLY with the title of the movie that best matches the question.
- **DO NOT include the year or any other information in your response.**
[/response_criteria]
```

#### Expected Impact

With this single change, the responses for "Jaws" and "Titanic" would almost certainly become exact matches. This would raise their scores from `0.6` to `1.0`.

* **Original Score Calculation:** (1 + 0 + 0.6 + 1 + 0.6) / 5 = **0.64**
* **Projected Score Calculation:** (1 + 0 + 1 + 1 + 1) / 5 = **0.80**

This demonstrates how a small, targeted change in the prompt can lead to a significant and measurable improvement in the system's performance, highlighting the power of iterative evaluation.

---

### **Appendix: Further Considerations for Evaluation and Debugging**

Building and testing a RAG system often raises more questions than it answers. Here are some deeper considerations inspired by our test results.

#### 1. Is the "Expected" Output Always Correct?

A key question to ask is whether "Jaws" is truly a better answer than "Jaws (1975)". For a user, the more specific answer might be more valuable. Our evaluation penalized the model for providing extra, correct information.

This reveals a crucial concept in automated evaluation: **Your evaluation is only as good as your test data and scoring criteria.**

* **Actionable Insight:** If the year is valuable, the `expected` field in `TEST_DATASET` should be updated to include it. If not, our prompt engineering fix is the correct path. This decision directly shapes the desired behavior of the AI.

#### 2. Debugging the "Back to the Future" Failure

Our prompt engineering tweak would not fix the test case that scored `0`, where the system responded with "Back to the Future Part II". This is a genuine failure that could stem from two primary causes:

1.  **Retrieval Failure:** The `$vectorSearch` query may have retrieved documents for the sequel as being more semantically similar to the prompt than documents for the original film.
2.  **Generation Failure:** The LLM may have ignored the provided context (which might have been correct) and relied on its own internal, and in this case flawed, knowledge.

**How to investigate this?**
The first step is to **log the retrieved context**. Modify the `run_rag_task` function to print the `context_str` variable. This allows you to see exactly what information the LLM was given.
* If the context was about "Part II," then the retrieval step is the problem. You might need to adjust `$vectorSearch` parameters like `numCandidates` or even consider if a different embedding model would create better vector representations for your data.
* If the context was correct (about the first movie), then the LLM is at fault. You would need to strengthen the prompt further, perhaps by explicitly commanding it to **ignore any prior knowledge and use only the provided context.**

#### 3. The Importance of Structured Responses

The current script uses `response_format={"type": "json_object"}` and asks the model to return a JSON object with specific fields: `response` and `used_provided_context`.

This is an excellent practice for several reasons:
* **Reliability:** It makes parsing the model's output robust. You are no longer trying to parse a free-form string.
* **Metadata:** You can ask the model for metadata about its own process. While the `used_provided_context` field is a self-report and not a guarantee, logging it can provide another signal for debugging. If the model consistently says it's not using the context for failed answers, it points toward a generation failure.
