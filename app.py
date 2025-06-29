from flask import Flask, request, render_template, redirect, url_for, session  
import os  
import sys  
import copy  
from pymongo import MongoClient  
from autoevals import Factuality, LLMClassifier, init  
from openai import AzureOpenAI  
import json  
from datetime import datetime  
from dotenv import load_dotenv  
from bson.objectid import ObjectId  
import time  # For timing  
from autoevals.ragas import ContextRelevancy, Faithfulness  # Import new evaluators  
  
# --- Configuration ---  
# Load environment variables  
load_dotenv()  
  
# Set a secret key for session management  
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key")  
app = Flask(__name__)  
app.secret_key = SECRET_KEY  
  
# Azure OpenAI Configuration  
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")  
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")  
AZURE_API_VERSION = "2024-12-01-preview"  # Update to your API version if necessary  
  
# Deployment names available for selection  
DEPLOYMENT_NAMES = ["o3-mini", "gpt-4o", "gpt-4o-mini"]  # List of available deployment names  
  
# Default deployment name  
DEFAULT_DEPLOYMENT_NAME = "o3-mini"  
  
# Azure Embedding Deployment Name  
AZURE_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"  # Replace with your embedding model deployment  
  
# MongoDB Configuration  
MONGO_URI = os.environ.get("MONGO_URI")  # Ensure this is set in your environment variables  
  
DB_NAME = "mdb_autoevals"  # Database name  
  
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
    print("Successfully initialized AzureOpenAI client.")  
except Exception as e:  
    print(f"Fatal Error: Could not initialize AzureOpenAI client. Check your configuration. Details: {e}",  
          file=sys.stderr)  
    sys.exit(1)  
  
# Initialize MongoDB Client  
mongo_client = None  
if MONGO_URI:  
    try:  
        mongo_client = MongoClient(MONGO_URI)  
        db = mongo_client[DB_NAME]  
        test_runs_collection = db["test_runs"]  
        print(f"Successfully connected to MongoDB '{DB_NAME}' database.")  
    except Exception as e:  
        print(f"Fatal Error: Could not connect to MongoDB. Check MONGO_URI. Details: {e}", file=sys.stderr)  
        sys.exit(1)  
else:  
    print(  
        "Warning: MONGO_URI is not set. The RAG task will not be able to query the database.",  
        file=sys.stderr,  
    )  
  
# --- Metric Definitions and Initialization ---  
# Define custom scorer function  
def exact_match(output, expected, **kwargs):  
    score = 1 if output.strip().lower() == expected.strip().lower() else 0  
    reason = "Exact match" if score == 1 else "Output does not match expected output"  
    metadata = {'expected': expected, 'output': output}  
    # Wrap score in an object with 'score', 'reason', 'metadata' attributes  
    return type('Result', (object,), {'score': score, 'reason': reason, 'metadata': metadata})  
  
# Initialize instances of metric evaluators  
factuality_evaluator = Factuality(client=azure_client)  # Pass the Azure client  
kid_friendly_classifier = LLMClassifier(  
    name="Kid-Friendly Movie Classifier",  
    prompt_template=(  
        "On a scale of 0 to 1, where 1 is perfectly suitable for children under 13 years old and 0 is completely unsuitable, "  
        "how kid-friendly is the following movie description? "  
        "Choose one of the following categories that best represents the score:\n"  
        "- 'Category_0.0' (Completely Unsuitable - score 0.0-0.1)\n"  
        "- 'Category_0.25' (Mostly Unsuitable - score 0.1-0.4)\n"  
        "- 'Category_0.5' (Moderately Kid-Friendly - score 0.4-0.6)\n"  
        "- 'Category_0.75' (Very Kid-Friendly - score 0.6-0.9)\n"  
        "- 'Category_1.0' (Perfectly Suitable - score 0.9-1.0)\n\n"  
        "Movie Description: {{output}}"  
    ),  
    choice_scores={  
        "Category_0.0": 0.0,  
        "Category_0.25": 0.25,  
        "Category_0.5": 0.5,  
        "Category_0.75": 0.75,  
        "Category_1.0": 1.0,  
    },  
    use_cot=True,  
    client=azure_client  # Pass the Azure client  
)  
  
# Instantiate new evaluators  
relevancy_evaluator = ContextRelevancy(client=azure_client)  
faithfulness_evaluator = Faithfulness(client=azure_client)  
  
# Define available metrics  
METRICS = {  
    'Factuality': factuality_evaluator,     # Instance of Factuality evaluator  
    'Exact Match': exact_match,             # Custom exact match scorer (function)  
    'Kid Friendly': kid_friendly_classifier,    # Instance of LLMClassifier  
    'Context Relevancy': relevancy_evaluator,   # Instance of ContextRelevancy  
    'Faithfulness': faithfulness_evaluator,     # Instance of Faithfulness  
}  
  
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
    """  
    if not mongo_client:  
        print("Cannot perform vector search, MongoDB client not initialized.", file=sys.stderr)  
        return []  
  
    pipeline = [  
        {  
            "$vectorSearch": {  
                "index": "embeddings_1_search_index",  
                "path": "plot_embedding",  # The field in your documents that contains the vector  
                "queryVector": vector,  
                "numCandidates": 200,  # Number of candidates to consider  
                "limit": 5,            # Number of results to return  
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
        },  
    ]  
    try:  
        results = list(mongo_client["sample_mflix"]["embedded_movies"].aggregate(pipeline))  
        return results  
    except Exception as e:  
        print(f"Error during vector search in MongoDB: {e}", file=sys.stderr)  
        return []  
  
def run_rag_task(input_prompt: str, deployment_name: str, response_criteria: str, system_prompt_template: str, user_prompt_template: str):  
    """  
    Executes the full Retrieval-Augmented Generation (RAG) task:  
    1. Generates an embedding for the input.  
    2. Retrieves context from MongoDB via vector search.  
    3. Generates a final response using the retrieved context.  
    Returns the generated response, the messages sent to the Azure client, and the context documents.  
    """  
    if not azure_client:  
        return "Error: Azure OpenAI client is not initialized.", [], []  
    if not mongo_client:  
        return "Error: MongoDB client is not initialized.", [], []  
  
    # 1. Get query vector  
    print(f"Generating embedding for query: '{input_prompt}'")  
    query_vector = get_embedding(input_prompt)  
    if not query_vector:  
        return "Error: Failed to generate embedding for the query.", [], []  
  
    # 2. Query MongoDB for context  
    print("Performing vector search in MongoDB...")  
    context_docs = perform_vector_search(query_vector)  
    if not context_docs:  
        print("No context found from vector search.", file=sys.stderr)  
        # Fallback: provide default context or handle as needed  
        context_str = "No specific context was found."  
    else:  
        print(f"Retrieved {len(context_docs)} documents from MongoDB.")  
        # Format the context for the prompt  
        context_str = "\n".join(  
            [f"- {doc['title']} ({doc['year']})\n\n{doc['plot']}" for doc in context_docs]  
        )  
  
    # 3. Generate response from the chat model with the retrieved context  
  
    try:  
        # Format the system and user prompts with the provided templates  
        system_prompt = system_prompt_template.replace("{context}", context_str)  
        user_prompt = user_prompt_template.replace("{response_criteria}", response_criteria).replace("{question}", input_prompt)  
  
        messages = [  
            {  
                "role": "system",  
                "content": system_prompt,  
            },  
            {  
                "role": "user",  
                "content": user_prompt,  
            },  
        ]  
  
        response = azure_client.chat.completions.create(  
            model=deployment_name,  
            messages=messages,  
            stream=False  
        )  
        generated_response = response.choices[0].message.content.strip()  
        return generated_response, messages, context_docs  # Return the context_docs as well  
    except Exception as e:  
        print(f"An unexpected error occurred during chat completion: {e}", file=sys.stderr)  
        return "", [], context_docs  # Return empty response and messages on error, but pass context_docs  
  
# Test Dataset  
TEST_DATASET = [  
    {  
        "id": 1,  
        "input": "A boy befriends a giant robot from outer space. \nWhat is the movie?",  
        "expected": "The Iron Giant"  
    },  
    {  
        "id": 2,  
        "input": "An 8-year-old boy genius and his friends must rescue their parents. \nWhat is the movie?",  
        "expected": "Jimmy Neutron: Boy Genius"  
    },  
    {  
        "id": 3,  
        "input": "An isolated research facility becomes the battleground as a trio of intelligent sharks fight back. \nWhat is the movie?",  
        "expected": "Deep Blue Sea"  
    },  
    {  
        "id": 4,  
        "input": "A hacker has to choose between a red pill and a blue pill to see the true nature of reality. \nWhat is the movie?",  
        "expected": "The Matrix"  
    },  
    {  
        "id": 5,  
        "input": "When two kids find and play a magical board game, they release a man trapped for decades and a host of dangers that can only be stopped by finishing the game. \nWhat is the movie?",  
        "expected": "Jumanji"  
    },  
]  
  
def get_test_dataset_from_form(form_data):  
    """Extract test cases from form data."""  
    test_case_count = int(form_data.get('test_case_count', 0))  
    test_dataset = []  
    for idx in range(test_case_count):  
        input_prompt = form_data.get(f'input_prompt_{idx}', '').strip()  
        expected_output = form_data.get(f'expected_output_{idx}', '').strip()  
        if input_prompt:  
            test_case = {  
                'id': idx + 1,  
                'input': input_prompt,  
                'expected': expected_output  
            }  
            test_dataset.append(test_case)  
    return test_dataset  
  
def test_rag_task_with_metrics(test_dataset, deployment_names, response_criteria, system_prompt_template, user_prompt_template, selected_metrics):  
    """Runs the RAG task on test data for multiple deployment names, evaluates selected metrics, and returns the results."""  
    test_run = {  
        "timestamp": datetime.utcnow().isoformat(),  
        "deployment_names": deployment_names,  # Note the plural key  
        "response_criteria": response_criteria,  
        "system_prompt_template": system_prompt_template,  
        "user_prompt_template": user_prompt_template,  
        "selected_metrics": selected_metrics,  
        "test_cases": [],  # We'll store results here  
        "models": {},  # Store results per model  
        "total_duration_seconds": 0,  
    }  
  
    # Start overall timing  
    overall_start_time = time.perf_counter()  
  
    # Loop over each deployment name (model)  
    for deployment_name in deployment_names:  
        print(f"Running tests for deployment: {deployment_name}")  
        # Start timing for this deployment  
        start_time = time.perf_counter()  
  
        # Initialize total scores per metric  
        total_scores = {metric: 0 for metric in selected_metrics}  
        model_test_cases = []  
        for idx, test_case in enumerate(test_dataset):  
            input_prompt = test_case["input"]  
            expected_output = test_case["expected"].strip()  
  
            generated_output, messages, context_docs = run_rag_task(  
                input_prompt,  
                deployment_name,  
                response_criteria,  
                system_prompt_template,  
                user_prompt_template  
            )  
  
            # Reconstruct context_str from context_docs  
            if context_docs:  
                context_str = "\n".join(  
                    [f"- {doc['title']} ({doc['year']})\n\n{doc['plot']}" for doc in context_docs]  
                )  
            else:  
                context_str = "No specific context was found."  
  
            metric_results = {}  
  
            for metric_name in selected_metrics:  
                evaluator = METRICS.get(metric_name)  
                if not evaluator:  
                    print(f"Evaluator for {metric_name} is not initialized.", file=sys.stderr)  
                    continue  
  
                # Prepare inputs for each metric  
                if callable(evaluator) and not hasattr(evaluator, 'eval'):  
                    # For function-based metrics like exact_match  
                    result = evaluator(  
                        output=generated_output,  
                        expected=expected_output  
                    )  
                elif hasattr(evaluator, 'eval'):  
                    # For class-based evaluators with an eval method  
                    if metric_name == "Factuality":  
                        # Ensure that 'expected_output' is provided for Factuality evaluation  
                        if not expected_output:  
                            print(f"Expected output is required for Factuality evaluation in Test Case {idx+1}.", file=sys.stderr)  
                            result = type('Result', (object,), {'score': 0, 'reason': 'Expected output is missing.', 'metadata': {}})  
                        else:  
                            result = evaluator.eval(  
                                input=input_prompt,  
                                output=generated_output,  
                                expected=expected_output  
                            )  
                    elif metric_name == "Kid Friendly":  
                        result = evaluator.eval(  
                            output=generated_output  
                        )  
                    elif metric_name in ("Context Relevancy", "Faithfulness"):  
                        result = evaluator.eval(  
                            input=input_prompt,  
                            output=generated_output,  
                            context=context_str  
                        )  
                    else:  
                        print(f"Metric {metric_name} is not recognized.", file=sys.stderr)  
                        continue  
                else:  
                    print(f"Evaluator for {metric_name} is not callable or does not have an eval method.", file=sys.stderr)  
                    continue  
  
                # Collect 'score', 'reason', 'metadata'  
                metric_results[metric_name] = {  
                    'score': result.score,  
                    'reason': getattr(result, 'reason', ''),  
                    'metadata': getattr(result, 'metadata', {}),  
                }  
                total_scores[metric_name] += result.score  
  
            test_case_result = {  
                "test_case_id": test_case.get("id", idx + 1),  
                "input_prompt": input_prompt,  
                "expected_output": expected_output,  
                "generated_output": generated_output,  
                "metric_results": metric_results,  
                "messages": messages,  # Store the messages sent to the Azure client  
                "context_docs": context_docs,  # Store the context documents  
                "deployment_name": deployment_name,  # Include deployment name in test case result  
            }  
  
            model_test_cases.append(test_case_result)  
  
            print(f"\n{'='*10} Test Case {idx+1} for {deployment_name} {'='*10}")  
            print(f"Input Prompt:\n{input_prompt}")  
            print(f"Expected Output:\n{expected_output}")  
            print(f"Generated Output:\n{generated_output}")  
            print(f"Metric Results:")  
            for m_name, res in metric_results.items():  
                print(f"- {m_name} Score: {res['score']}")  
                print(f"  Metadata: {res['metadata']}")  
            print(f"{'='*30}")  
  
        # Calculate average scores per metric for this model  
        average_scores = {}  
        for metric_name in total_scores:  
            average_score = total_scores[metric_name] / len(test_dataset) if test_dataset else 0  
            average_scores[metric_name] = average_score  
  
        # End timing for this deployment  
        end_time = time.perf_counter()  
        deployment_duration = end_time - start_time  
  
        # Store results per model, including duration  
        test_run["models"][deployment_name] = {  
            "average_scores": average_scores,  
            "test_cases": model_test_cases,  
            "duration_seconds": deployment_duration,  # Add duration per deployment  
        }  
  
        # Append to overall test cases  
        test_run["test_cases"].extend(model_test_cases)  
  
        print(f"\n--- Test Run Summary for {deployment_name} ---")  
        for metric_name, avg_score in average_scores.items():  
            print(f"Average {metric_name} Score: {avg_score}")  
  
    # End overall timing  
    overall_end_time = time.perf_counter()  
    total_duration = overall_end_time - overall_start_time  
    test_run["total_duration_seconds"] = total_duration  
  
    return test_run  
  
# --- Flask Routes ---  
  
@app.route('/', methods=['GET'])  
def index():  
    # Fetch previous test runs from MongoDB  
    previous_runs = list(  
        test_runs_collection.find(  
            {},  
            {  
                "timestamp": 1,  
                "deployment_names": 1,  
                "deployment_name": 1,  # Include for older test runs  
                "models": 1,  # Include models to access durations  
                "average_scores": 1,  
                "response_criteria": 1,  
                "system_prompt_template": 1,  
                "user_prompt_template": 1,  
                "test_cases.input_prompt": 1,  
                "selected_metrics": 1,  
                "total_duration_seconds": 1,  # Include the total duration  
            }  
        ).sort("timestamp", -1).limit(10)  
    )  # Get last 10 runs  
  
    # Use the test dataset from the session if available  
    if 'test_dataset' in session:  
        test_dataset = session['test_dataset']  
    else:  
        test_dataset = copy.deepcopy(TEST_DATASET)  
  
    default_response_criteria = """  
- Provide a concise answer to the question based ONLY on the context.  
- Respond ONLY with the complete title of the movie that best matches the question.  
    """  
    default_system_prompt = """  
Below is the context, which includes plots of movies.  
  
[context]  
{context}  
[/context]  
    """  
    default_user_prompt = """  
[response_criteria]  
{response_criteria}  
[/response_criteria]  
  
[question]  
{question}  
[/question]  
    """  
  
    # Prepare the list of available metrics and mark Factuality as selected by default  
    available_metrics = [{'name': name, 'selected': (name == 'Factuality')} for name in METRICS.keys()]  
  
    return render_template(  
        'index.html',  
        test_dataset=test_dataset,  
        deployment_names=DEPLOYMENT_NAMES,  
        default_deployment=DEFAULT_DEPLOYMENT_NAME,  
        previous_runs=previous_runs,  
        default_response_criteria=default_response_criteria.strip(),  
        default_system_prompt=default_system_prompt.strip(),  
        default_user_prompt=default_user_prompt.strip(),  
        current_year=datetime.utcnow().year,  # Pass current_year to template  
        available_metrics=available_metrics  
    )  
  
@app.route('/add_test_case', methods=['POST'])  
def add_test_case():  
    # Get the test case data from the form  
    input_prompt = request.form.get('new_input_prompt', '').strip()  
    expected_output = request.form.get('new_expected_output', '').strip()  
  
    # Retrieve the current test dataset from the session or default  
    if 'test_dataset' in session:  
        test_dataset = session['test_dataset']  
    else:  
        test_dataset = copy.deepcopy(TEST_DATASET)  
  
    if input_prompt:  
        new_test_case = {  
            'id': len(test_dataset) + 1,  
            'input': input_prompt,  
            'expected': expected_output  
        }  
  
        test_dataset.append(new_test_case)  
        session['test_dataset'] = test_dataset  # Update the session  
  
    return redirect(url_for('index'))  
  
@app.route('/remove_test_case', methods=['POST'])  
def remove_test_case():  
    # Get the test case id to remove  
    test_case_id = int(request.form.get('test_case_id_to_remove', '-1'))  
  
    if 'test_dataset' in session:  
        test_dataset = session['test_dataset']  
    else:  
        test_dataset = copy.deepcopy(TEST_DATASET)  
  
    # Remove the test case with the matching id  
    test_dataset = [tc for tc in test_dataset if tc['id'] != test_case_id]  
  
    # Renumber the ids  
    for idx, tc in enumerate(test_dataset, start=1):  
        tc['id'] = idx  
  
    session['test_dataset'] = test_dataset  # Update the session  
  
    return redirect(url_for('index'))  
  
@app.route('/run_test', methods=['POST'])  
def run_test():  
    # Extract test cases from form data  
    test_dataset = get_test_dataset_from_form(request.form)  
    # Retrieve the selected deployment names from the form  
    deployment_names = request.form.getlist('deployment_names')  
    if not deployment_names:  
        # Default to the default deployment name if none are selected  
        deployment_names = [DEFAULT_DEPLOYMENT_NAME]  
    # Get the response criteria from the form  
    response_criteria = request.form.get('response_criteria', '').strip()  
    # Get system and user prompt templates  
    system_prompt_template = request.form.get('system_prompt_template', '').strip()  
    user_prompt_template = request.form.get('user_prompt_template', '').strip()  
    # Get selected metrics  
    selected_metrics = request.form.getlist('metrics')  
    if not selected_metrics:  
        # Default to Factuality if no metrics are selected  
        selected_metrics = ['Factuality']  
    # Run the test  
    test_run_data = test_rag_task_with_metrics(  
        test_dataset,  
        deployment_names,  
        response_criteria,  
        system_prompt_template,  
        user_prompt_template,  
        selected_metrics  
    )  
  
    # Save test run data to MongoDB  
    inserted_id = test_runs_collection.insert_one(test_run_data).inserted_id  
    test_run_data['_id'] = inserted_id  # Include the ID for rendering  
  
    return render_template('test_results.html', test_run=test_run_data, current_year=datetime.utcnow().year)  
  
@app.route('/preview', methods=['POST'])  
def preview():  
    # Get the index of the test case to preview  
    idx = int(request.form.get('test_case_idx', -1))  
    if idx < 0:  
        return "Error: Invalid test case index.", 400  
  
    # Directly extract the input prompt and expected output for the specific test case  
    input_prompt = request.form.get(f'input_prompt_{idx}', '').strip()  
    expected_output = request.form.get(f'expected_output_{idx}', '').strip()  
    if not input_prompt:  
        return "Error: Input prompt is empty.", 400  
  
    # Retrieve the selected deployment names from the form  
    deployment_names = request.form.getlist('deployment_names')  
    if not deployment_names:  
        # Default to the default deployment name if none are selected  
        deployment_names = [DEFAULT_DEPLOYMENT_NAME]  
    # Get the response criteria from the form  
    response_criteria = request.form.get('response_criteria', '').strip()  
    # Get system and user prompt templates  
    system_prompt_template = request.form.get('system_prompt_template', '').strip()  
    user_prompt_template = request.form.get('user_prompt_template', '').strip()  
    # Get selected metrics  
    selected_metrics = request.form.getlist('metrics')  
    if not selected_metrics:  
        # Default to Factuality if no metrics are selected  
        selected_metrics = ['Factuality']  
  
    # We will just run the preview for the first selected deployment name  
    deployment_name = deployment_names[0]  
  
    generated_output, messages, context_docs = run_rag_task(input_prompt, deployment_name, response_criteria, system_prompt_template, user_prompt_template)  
  
    # Reconstruct context_str from context_docs  
    if context_docs:  
        context_str = "\n".join(  
            [f"- {doc['title']} ({doc['year']})\n\n{doc['plot']}" for doc in context_docs]  
        )  
    else:  
        context_str = "No specific context was found."  
  
    metric_results = {}  
  
    for metric_name in selected_metrics:  
        evaluator = METRICS.get(metric_name)  
        if not evaluator:  
            print(f"Evaluator for {metric_name} is not initialized.", file=sys.stderr)  
            continue  
  
        # Prepare inputs for each metric  
        if callable(evaluator) and not hasattr(evaluator, 'eval'):  
            # For function-based metrics like exact_match  
            result = evaluator(  
                output=generated_output,  
                expected=expected_output  
            )  
        elif hasattr(evaluator, 'eval'):  
            # For class-based evaluators with an eval method  
            if metric_name == "Factuality":  
                # Ensure that 'expected_output' is provided for Factuality evaluation  
                if not expected_output:  
                    print(f"Expected output is required for Factuality evaluation in preview.", file=sys.stderr)  
                    result = type('Result', (object,), {'score': 0, 'reason': 'Expected output is missing.', 'metadata': {}})  
                else:  
                    result = evaluator.eval(  
                        input=input_prompt,  
                        output=generated_output,  
                        expected=expected_output  
                    )  
            elif metric_name == "Kid Friendly":  
                result = evaluator.eval(  
                    output=generated_output  
                )  
            elif metric_name in ("Context Relevancy", "Faithfulness"):  
                result = evaluator.eval(  
                    input=input_prompt,  
                    output=generated_output,  
                    context=context_str  
                )  
            else:  
                print(f"Metric {metric_name} is not recognized.", file=sys.stderr)  
                continue  
        else:  
            print(f"Evaluator for {metric_name} is not callable or does not have an eval method.", file=sys.stderr)  
            continue  
  
        # Collect 'score', 'reason', 'metadata'  
        metric_results[metric_name] = {  
            'score': result.score,  
            'reason': getattr(result, 'reason', ''),  
            'metadata': getattr(result, 'metadata', {}),  
        }  
  
    test_case_result = {  
        "input_prompt": input_prompt,  
        "expected_output": expected_output,  
        "generated_output": generated_output,  
        "metric_results": metric_results,  
        "deployment_name": deployment_name,  
        "response_criteria": response_criteria,  
        "system_prompt_template": system_prompt_template,  
        "user_prompt_template": user_prompt_template,  
        "messages": messages,  # Store the messages  
        "context_docs": context_docs,  # Include the context documents  
    }  
    # Render only the content  
    return render_template('preview_content.html', test_case=test_case_result)  
  
@app.route('/test_runs', methods=['GET'])  
def test_runs():  
    # Fetch all test runs  
    runs = list(  
        test_runs_collection.find(  
            {},  
            {  
                "timestamp": 1,  
                "deployment_names": 1,  
                "deployment_name":1, # Include for older test runs  
                "models": 1,  # Include models to access durations  
                "average_scores": 1,  
                "response_criteria": 1,  
                "system_prompt_template": 1,  
                "user_prompt_template": 1,  
                "test_cases.input_prompt": 1,  
                "selected_metrics": 1,  
                "total_duration_seconds": 1,  # Include the total duration  
            }  
        ).sort("timestamp", -1)  
    )  
    return render_template('test_run_list.html', test_runs=runs, current_year=datetime.utcnow().year)  
  
@app.route('/test_run/<string:run_id>', methods=['GET'])  
def view_test_run(run_id):  
    # Retrieve test run data from MongoDB using the run_id  
    test_run = test_runs_collection.find_one({"_id": ObjectId(run_id)})  
    if not test_run:  
        return "Test run not found."  
    return render_template('test_results.html', test_run=test_run, current_year=datetime.utcnow().year)  
  
@app.route('/reset', methods=['GET'])  
def reset():  
    # Clear the test_dataset from the session  
    session.pop('test_dataset', None)  
    # Redirect to home, defaults will be loaded  
    return redirect(url_for('index'))  
  
if __name__ == '__main__':  
    app.run(debug=True)  