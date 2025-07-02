from flask import Flask, request, render_template, redirect, url_for, session, jsonify      
import os      
import sys      
import copy      
import re      
from typing import Optional      
import pymongo      
from pymongo import MongoClient      
from pymongo.errors import PyMongoError, OperationFailure      
from pymongo.operations import SearchIndexModel      
import json      
import datetime      
from dotenv import load_dotenv      
from bson.objectid import ObjectId      
import time      
import bson      
import decimal      
import logging      
from autoevals import Factuality, LLMClassifier, init      
from autoevals.ragas import ContextRelevancy, Faithfulness      
      
class MDBAutoEval:      
    def __init__(self):      
        # Load environment variables      
        load_dotenv()      
      
        # Configure logging      
        logging.basicConfig(level=logging.INFO)      
        self.logger = logging.getLogger(__name__)      
      
        # Set a secret key for session management      
        SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key")      
        self.app = Flask(__name__)      
        self.app.secret_key = SECRET_KEY      
      
        # Azure OpenAI Configuration      
        self.AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")      
        self.AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")      
        self.AZURE_API_VERSION = "2024-12-01-preview"  # Update to your API version if necessary      
      
        # VoyageAI Configuration      
        self.VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")  # Ensure this is set in your environment variables      
      
        # Deployment names available for selection      
        self.DEPLOYMENT_NAMES = ["o3-mini", "gpt-4o", "gpt-4o-mini"]  # List of available deployment names      
      
        # Default deployment name      
        self.DEFAULT_DEPLOYMENT_NAME = "o3-mini"      
      
        # Embedding Models Configuration      
        # Define your embedding models, their providers, and necessary details      
        self.EMBEDDING_MODELS = {      
            # Azure OpenAI Embedding Models      
            "text-embedding-3-large": {      
                "deployment_name": "text-embedding-3-large",      
                "provider": "azure_openai"      
            },      
            "text-embedding-3-small": {      
                "deployment_name": "text-embedding-3-small",      
                "provider": "azure_openai"      
            },      
            "text-embedding-ada-002": {      
                "deployment_name": "text-embedding-ada-002",      
                "provider": "azure_openai"      
            },      
            # VoyageAI Embedding Models      
            "voyage-3-large": {      
                "model_name": "voyage-3-large",      
                "provider": "voyageai"      
            },      
            "voyage-3.5": {      
                "model_name": "voyage-3.5",      
                "provider": "voyageai"      
            },      
            "voyage-3.5-lite": {      
                "model_name": "voyage-3.5-lite",      
                "provider": "voyageai"      
            },      
        }      
      
        # Default embedding model name      
        self.DEFAULT_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"      
      
        # MongoDB Configuration      
        self.MONGO_URI = os.environ.get("MONGO_URI")  # Ensure this is set in your environment variables      
        self.DB_NAME = "mdb_autoevals"  # Database name      
      
        # Initialize clients      
        self.azure_client = None      
        self.voyage_client = None      
        self.mongo_client = None      
        self.db = None      
        self.test_runs_collection = None      
        self.evaluators_collection = None      
        self.idx_meta_collection = None      
      
        # Evaluation Metrics      
        self.METRICS = {}      
        self.TEST_DATASET = []  # Now we remove the hardcoded TEST_DATASET      
      
        self.initialize_clients()      
        self.init_evaluators()      
      
    def bson_serializer(self, obj):      
        if isinstance(obj, datetime.datetime):      
            return obj.isoformat()      
        elif isinstance(obj, datetime.date):      
            return obj.isoformat()      
        elif isinstance(obj, datetime.time):      
            return obj.isoformat()      
        elif isinstance(obj, ObjectId):      
            return str(obj)      
        elif isinstance(obj, bytes):      
            try:      
                return obj.decode('utf-8')      
            except UnicodeDecodeError:      
                return repr(obj)      
        elif isinstance(obj, decimal.Decimal):      
            return float(obj)      
        else:      
            return str(obj)      
      
    def initialize_clients(self):      
        # Initialize Azure OpenAI Client      
        try:      
            if not self.AZURE_ENDPOINT or not self.AZURE_API_KEY:      
                raise ValueError("Azure endpoint or API key is not configured. Please set the environment variables.")      
      
            from openai import AzureOpenAI      
            self.azure_client = AzureOpenAI(      
                api_version=self.AZURE_API_VERSION,      
                azure_endpoint=self.AZURE_ENDPOINT,      
                api_key=self.AZURE_API_KEY,      
            )      
            self.logger.info("Successfully initialized AzureOpenAI client.")      
        except Exception as e:      
            self.logger.error(f"Fatal Error: Could not initialize AzureOpenAI client. Check your configuration. Details: {e}")      
            sys.exit(1)      
      
        # Initialize VoyageAI Client      
        try:      
            import voyageai      
            self.voyage_client = voyageai.Client(api_key=self.VOYAGE_API_KEY)      
            self.logger.info("Successfully initialized VoyageAI client.")      
        except Exception as e:      
            self.logger.error(f"Fatal Error: Could not initialize VoyageAI client. Check your configuration. Details: {e}")      
      
        # Initialize MongoDB Client      
        if self.MONGO_URI:      
            try:      
                self.mongo_client = MongoClient(self.MONGO_URI)      
                self.db = self.mongo_client[self.DB_NAME]      
                self.test_runs_collection = self.db["test_runs"]      
                self.evaluators_collection = self.db["evaluators"]  # Unified collection for all evaluators      
                self.idx_meta_collection = self.db["idx_meta"]  # Existing collection to store index metadata      
                self.logger.info(f"Successfully connected to MongoDB '{self.DB_NAME}' database.")      
            except Exception as e:      
                self.logger.error(f"Fatal Error: Could not connect to MongoDB. Check MONGO_URI. Details: {e}")      
                sys.exit(1)      
      
    def exact_match(self, output, expected, **kwargs):      
        score = 1 if output.strip().lower() == expected.strip().lower() else 0      
        reason = "Exact match" if score == 1 else "Output does not match expected output"      
        metadata = {'expected': expected, 'output': output}      
        return type('Result', (object,), {'score': score, 'reason': reason, 'metadata': metadata})      
      
    def init_evaluators(self):      
      
        # Initialize autoevals with the Azure client      
        init(self.azure_client)      
      
        self.factuality_evaluator = Factuality(client=self.azure_client)  # Pass the Azure client      
      
        # Instantiate new evaluators      
        self.relevancy_evaluator = ContextRelevancy(client=self.azure_client)      
        self.faithfulness_evaluator = Faithfulness(client=self.azure_client)      
      
        # Define available metrics      
        self.METRICS = {      
            'Factuality': self.factuality_evaluator,     # Instance of Factuality evaluator      
            'Exact Match': self.exact_match,             # Custom exact match scorer (function)      
            'Context Relevancy': self.relevancy_evaluator,   # Instance of ContextRelevancy      
            'Faithfulness': self.faithfulness_evaluator,     # Instance of Faithfulness      
        }      
        self.load_evaluators_from_db()      
      
    def load_evaluators_from_db(self):      
        try:      
            evaluators_cursor = self.evaluators_collection.find({})      
            for eval_doc in evaluators_cursor:      
                name = eval_doc.get('name')      
                if name in self.METRICS:      
                    self.logger.info(f"Evaluator '{name}' already exists in METRICS, skipping...")      
                    continue      
                evaluator_type = eval_doc.get('type', 'LLMClassifier')  # Default to LLMClassifier if not specified      
      
                if evaluator_type == 'LLMClassifier':      
                    # Load LLM-based classifier      
                    prompt_template = eval_doc.get('prompt_template')      
                    choice_scores = eval_doc.get('choice_scores')      
                    use_cot = eval_doc.get('use_cot', False)      
                    model_deployment_name = eval_doc.get('model_deployment_name', self.DEFAULT_DEPLOYMENT_NAME)      
                    temperature = eval_doc.get('temperature', 0.0)      
      
                    # Instantiate an LLMClassifier with these parameters      
                    classifier_instance = LLMClassifier(      
                        name=name,      
                        prompt_template=prompt_template,      
                        choice_scores=choice_scores,      
                        use_cot=use_cot,      
                        client=self.azure_client,      
                        deployment_name=model_deployment_name,      
                        temperature=temperature      
                    )      
      
                    # Add the classifier to the METRICS dictionary      
                    self.METRICS[name] = classifier_instance      
                    self.logger.info(f"Loaded LLMClassifier evaluator '{name}' from database.")      
                elif evaluator_type == 'FunctionEvaluator':      
                    # Load function-based evaluator      
                    function_subtype = eval_doc.get('function_subtype')      
                    parameters = eval_doc.get('parameters', {})      
      
                    if function_subtype == 'regex_match':      
                        pattern = parameters.get('pattern')      
                        if pattern:      
                            def make_regex_evaluator(pattern):      
                                def evaluator(output, expected, **kwargs):      
                                    import re      
                                    match = re.search(pattern, output)      
                                    score = 1 if match else 0      
                                    reason = "Pattern matched" if match else "Pattern did not match"      
                                    metadata = {'pattern': pattern, 'output': output}      
                                    return type('Result', (object,), {'score': score, 'reason': reason, 'metadata': metadata})      
                                return evaluator      
                            evaluator = make_regex_evaluator(pattern)      
                            self.METRICS[name] = evaluator      
                            self.logger.info(f"Added function-based evaluator '{name}' of subtype 'regex_match' to METRICS")      
                        else:      
                            self.logger.warning(f"Pattern not provided for evaluator '{name}'. Skipping.")      
                    elif function_subtype == 'exact_match':      
                        # Use existing exact_match function      
                        self.METRICS[name] = self.exact_match      
                        self.logger.info(f"Added function-based evaluator '{name}' of subtype 'exact_match' to METRICS")      
                    else:      
                        self.logger.warning(f"Unknown function_subtype '{function_subtype}' for evaluator '{name}'. Skipping.")      
                else:      
                    self.logger.warning(f"Unknown evaluator type '{evaluator_type}' for evaluator '{name}'. Skipping.")      
        except Exception as e:      
            self.logger.exception("Error loading evaluators from database.")      
      
    def store_evaluator_metadata(self, evaluator_config):      
        """Stores or updates the evaluator metadata in the MongoDB 'evaluators' collection."""      
        try:      
            # Check if evaluator already exists based on unique name      
            existing_evaluator = self.evaluators_collection.find_one({"name": evaluator_config["name"]})      
            if existing_evaluator:      
                evaluator_id = existing_evaluator["_id"]      
                # Compare existing configuration with the new one      
                config_changed = False      
                for key in evaluator_config:      
                    if existing_evaluator.get(key) != evaluator_config[key]:      
                        config_changed = True      
                        break      
                if config_changed:      
                    # Update existing evaluator with new configuration      
                    self.evaluators_collection.update_one(      
                        {"_id": evaluator_id},      
                        {"$set": evaluator_config}      
                    )      
                    self.logger.info(f"Updated evaluator '{evaluator_config['name']}' with ID: {evaluator_id}")      
                else:      
                    self.logger.info(f"Evaluator '{evaluator_config['name']}' already exists with ID: {evaluator_id} and has the same configuration.")      
                return evaluator_id      
            else:      
                # Insert new evaluator metadata      
                result = self.evaluators_collection.insert_one(evaluator_config)      
                evaluator_id = result.inserted_id      
                self.logger.info(f"Stored new evaluator '{evaluator_config['name']}' with ID: {evaluator_id}")      
                return evaluator_id      
        except PyMongoError:      
            self.logger.exception("Failed to store evaluator metadata in MongoDB.")      
            return None      
      
    def to_pascal_case(self, text: str) -> str:      
        """Converts a string to PascalCase."""      
        return ''.join(word.capitalize() for word in re.split(r'[\W_]+', text))      
      
    def get_embedding(self, text: str, model_info: dict) -> Optional[list]:      
        """Generates a vector embedding for a given text using the specified model."""      
        provider = model_info.get('provider', 'azure_openai')  # Default to 'azure_openai' for backward compatibility      
        if provider == 'azure_openai':      
            model_deployment_name = model_info.get('deployment_name')      
            if not self.azure_client:      
                raise ValueError("Azure client is not initialized.")      
            try:      
                response = self.azure_client.embeddings.create(input=[text], model=model_deployment_name)      
                return response.data[0].embedding      
            except Exception as e:      
                self.logger.error(f"Error generating embedding: {e}")      
                return None      
        elif provider == 'voyageai':      
            model_name = model_info.get('model_name')      
            if not self.voyage_client:      
                raise ValueError("VoyageAI client is not initialized.")      
            try:      
                # VoyageAI embeddings code      
                response = self.voyage_client.embed(      
                    [text],      
                    model=model_name,      
                    input_type="document",      
                    output_dimension=model_info.get('output_dimension'),      
                    output_dtype=model_info.get('output_dtype', 'float')      
                )      
                embeddings = response.embeddings      
                return embeddings[0]      
            except Exception as e:      
                self.logger.error(f"Error generating VoyageAI embedding: {e}")      
                return None      
        else:      
            self.logger.error(f"Unknown embedding provider: {provider}")      
            return None      
      
    def perform_vector_search(self, vector: list, db_name: str, collection_name: str, index_name: str, vector_field: str, selected_fields: list) -> list:      
        """Performs a $search query in MongoDB to find relevant documents."""      
        if not self.mongo_client:      
            self.logger.error("Cannot perform vector search, MongoDB client not initialized.")      
            return []      
      
        # Build the vector search pipeline      
        pipeline = [      
            {      
                "$vectorSearch": {      
                    "index": index_name,      
                    "queryVector": vector,      
                    "path": vector_field,      
                    "numCandidates": 100,  # Adjust as needed for recall/latency tradeoff      
                    "limit": 5      
                }      
            },      
            {      
                "$project": {      
                    # Start with empty projection      
                    "_id": 0      
                }      
            }      
        ]      
      
        project_stage = pipeline[1]["$project"]      
      
        # Add the 'score' field from vector search      
        project_stage["vs_score"] = {"$meta": "vectorSearchScore"}      
      
        # Remove 'score' from selected_fields if present      
        selected_fields = [field for field in selected_fields if field != 'score']      
      
        # Add selected fields to the $project      
        for field in selected_fields:      
            project_stage[field] = 1      
      
        try:      
            collection = self.mongo_client[db_name][collection_name]      
      
            results = list(collection.aggregate(pipeline))      
            if not results:      
                self.logger.info("No documents found from vector search.")      
                return []      
      
            return results      
        except Exception as e:      
            self.logger.error(f"Error during vector search in MongoDB: {e}")      
            return []      
      
    def run_rag_task(self, input_prompt: str, deployment_name: str, response_criteria: str, system_prompt_template: str, user_prompt_template: str,      
                     db_name: str, collection_name: str, index_name: str, vector_field: str, embedding_model_name: str, selected_fields: list):      
        """Executes the full Retrieval-Augmented Generation (RAG) task."""      
        if not self.azure_client:      
            return "Error: Azure OpenAI client is not initialized.", [], []      
        if not self.mongo_client:      
            return "Error: MongoDB client is not initialized.", [], []      
      
        # Get embedding model info      
        embedding_model_info = self.EMBEDDING_MODELS.get(embedding_model_name)      
        if not embedding_model_info:      
            self.logger.error(f"Embedding model '{embedding_model_name}' is not recognized.")      
            return "Error: Embedding model not recognized.", [], []      
      
        # 1. Get query vector      
        self.logger.info(f"Generating embedding for query: '{input_prompt}' using embedding model '{embedding_model_name}'")      
        query_vector = self.get_embedding(input_prompt, embedding_model_info)      
        if not query_vector:      
            return "Error: Failed to generate embedding for the query.", [], []      
      
        # 2. Query MongoDB for context      
        self.logger.info("Performing vector search in MongoDB...")      
        context_docs = self.perform_vector_search(query_vector, db_name, collection_name, index_name, vector_field, selected_fields)      
        if not context_docs:      
            self.logger.info("No context found from vector search.")      
            # Fallback: provide default context or handle as needed      
            context_str = "No specific context was found."      
        else:      
            self.logger.info(f"Retrieved {len(context_docs)} documents from MongoDB.")      
            # Dynamically format the context for the prompt      
            context_str = "\n".join(      
                [json.dumps(doc, default=self.bson_serializer, indent=2) for doc in context_docs]      
            )      
      
            # Truncate context_str if it exceeds a certain length (e.g., 4096 characters)      
            if len(context_str) > 4000:      
                context_str = context_str[:4000]      
      
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
      
            response = self.azure_client.chat.completions.create(      
                model=deployment_name,      
                messages=messages,      
                stream=False      
            )      
            generated_response = response.choices[0].message.content.strip()      
            return generated_response, messages, context_docs  # Return the context_docs as well      
        except Exception as e:      
            self.logger.error(f"An unexpected error occurred during chat completion: {e}")      
            return "", [], context_docs  # Return empty response and messages on error, but pass context_docs      
      
    def get_test_dataset_from_form(self, form_data):      
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
      
    def test_rag_task_with_metrics(self, test_dataset, deployment_names, response_criteria, system_prompt_template, user_prompt_template, selected_metrics,      
                                   db_name, collection_name, index_name, vector_field, embedding_model_name, selected_fields):      
        """Runs the RAG task on test data for multiple deployment names, evaluates selected metrics, and returns the results."""      
        test_run = {      
            "timestamp": datetime.datetime.utcnow().isoformat(),      
            "deployment_names": deployment_names,  # Note the plural key      
            "response_criteria": response_criteria,      
            "system_prompt_template": system_prompt_template,      
            "user_prompt_template": user_prompt_template,      
            "selected_metrics": selected_metrics,      
            "test_cases": [],  # We'll store results here      
            "models": {},  # Store results per model      
            "total_duration_seconds": 0,      
            "db_name": db_name,      
            "collection_name": collection_name,      
            "index_name": index_name,      
            "vector_field": vector_field,      
            "embedding_model_name": embedding_model_name,  # Store the embedding model used      
            "selected_fields": selected_fields,  # Include selected fields      
        }      
      
        # Start overall timing      
        overall_start_time = time.perf_counter()      
      
        # Loop over each deployment name (model)      
        for deployment_name in deployment_names:      
            self.logger.info(f"Running tests for deployment: {deployment_name}")      
            # Start timing for this deployment      
            start_time = time.perf_counter()      
      
            # Initialize total scores per metric      
            total_scores = {metric: 0 for metric in selected_metrics}      
            model_test_cases = []      
            for idx, test_case in enumerate(test_dataset):      
                input_prompt = test_case["input"]      
                expected_output = test_case["expected"].strip()      
      
                generated_output, messages, context_docs = self.run_rag_task(      
                    input_prompt,      
                    deployment_name,      
                    response_criteria,      
                    system_prompt_template,      
                    user_prompt_template,      
                    db_name,      
                    collection_name,      
                    index_name,      
                    vector_field,      
                    embedding_model_name,      
                    selected_fields      
                )      
      
                # Reconstruct context_str from context_docs      
                if context_docs:      
                    context_str = "\n".join(      
                        [json.dumps(doc, default=self.bson_serializer, indent=2) for doc in context_docs]      
                    )      
                else:      
                    context_str = "No specific context was found."      
      
                metric_results = {}      
      
                for metric_name in selected_metrics:      
                    evaluator = self.METRICS.get(metric_name)      
                    if not evaluator:      
                        self.logger.error(f"Evaluator for {metric_name} is not initialized.")      
                        continue      
      
                    print(f"Evaluating {type(evaluator).__name__} for input: {input_prompt}, output: {generated_output}, context: {context_str}")      
      
                    # Prepare inputs for each metric      
                    if callable(evaluator) and not hasattr(evaluator, 'eval'):      
                        # For function-based metrics like exact_match      
                        result = evaluator(      
                            output=generated_output,      
                            expected=expected_output      
                        )      
                    elif hasattr(evaluator, 'eval'):      
                        # For class-based evaluators with an eval method      
                        if isinstance(evaluator, Factuality):      
                            # Ensure that 'expected_output' is provided for Factuality evaluation      
                            if not expected_output:      
                                self.logger.error(f"Expected output is required for Factuality evaluation in Test Case {idx+1}.")      
                                result = type('Result', (object,), {'score': 0, 'reason': 'Expected output is missing.', 'metadata': {}})      
                            else:      
                                result = evaluator.eval(      
                                    input=input_prompt,      
                                    output=generated_output,      
                                    expected=expected_output      
                                )      
                        elif isinstance(evaluator, LLMClassifier):      
                            result = evaluator.eval(      
                                output=generated_output      
                            )      
                        elif isinstance(evaluator, (ContextRelevancy, Faithfulness)):      
                            result = evaluator.eval(      
                                input=input_prompt,      
                                output=generated_output,      
                                context=context_str      
                            )      
                        else:      
                            self.logger.error(f"Evaluator of type {type(evaluator)} for metric {metric_name} is not recognized.")      
                            continue      
                    else:      
                        self.logger.error(f"Evaluator for {metric_name} is not callable or does not have an eval method.")      
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
      
                self.logger.info(f"\n{'='*10} Test Case {idx+1} for {deployment_name} {'='*10}")      
                self.logger.info(f"Input Prompt:\n{input_prompt}")      
                self.logger.info(f"Expected Output:\n{expected_output}")      
                self.logger.info(f"Generated Output:\n{generated_output}")      
                self.logger.info(f"Metric Results:")      
                for m_name, res in metric_results.items():      
                    self.logger.info(f"- {m_name} Score: {res['score']}")      
                    self.logger.info(f"  Metadata: {res['metadata']}")      
                self.logger.info(f"{'='*30}")      
      
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
      
            self.logger.info(f"\n--- Test Run Summary for {deployment_name} ---")      
            for metric_name, avg_score in average_scores.items():      
                self.logger.info(f"Average {metric_name} Score: {avg_score}")      
      
        # End overall timing      
        overall_end_time = time.perf_counter()      
        total_duration = overall_end_time - overall_start_time      
        test_run["total_duration_seconds"] = total_duration      
      
        return test_run      
      
    def get_databases(self):      
        """Retrieve a list of databases with collections, fields, and indexes."""      
        try:      
            db_names = self.mongo_client.list_database_names()      
            databases_info = []      
            for db_name in db_names:      
                if db_name in ('admin', 'local', 'config'):      
                    continue      
                db_info = {      
                    'name': db_name,      
                    'collections': []      
                }      
                collections = self.get_collections(db_name)      
                for collection_name in collections:      
                    fields = self.get_collection_fields(db_name, collection_name)      
                    indexes = self.get_atlas_search_indexes(db_name, collection_name)      
                    collection_info = {      
                        'name': collection_name,      
                        'fields': fields,      
                        'indexes': indexes      
                    }      
                    db_info['collections'].append(collection_info)      
                databases_info.append(db_info)      
            return databases_info      
        except PyMongoError as e:      
            self.logger.error(f"Error listing databases: {e}")      
            return []      
      
    def get_collections(self, db_name):      
        """Retrieve a list of collection names from a given database."""      
        try:      
            db = self.mongo_client[db_name]      
            return db.list_collection_names()      
        except PyMongoError as e:      
            self.logger.error(f"Error listing collections for database {db_name}: {e}")      
            return []      
      
    def get_collection_fields(self, db_name, collection_name):      
        """Retrieve a list of field names from a collection's sample documents."""      
        try:      
            collection = self.mongo_client[db_name][collection_name]      
            fields = set()      
            cursor = collection.find({}, limit=100)      
            for sample_doc in cursor:      
                self.recursive_extract_fields(sample_doc, fields)      
            return list(fields)      
        except Exception as e:      
            self.logger.error(f"Error fetching fields from {db_name}.{collection_name}: {e}")      
            return []      
      
    def recursive_extract_fields(self, document, fields, parent_prefix=''):      
        """Recursively extract field names from nested documents."""      
        for key, value in document.items():      
            field_full_name = f"{parent_prefix}.{key}" if parent_prefix else key      
            fields.add(field_full_name)      
            if isinstance(value, dict):      
                self.recursive_extract_fields(value, fields, field_full_name)      
      
    def get_atlas_search_indexes(self, db_name, collection_name):      
        """Retrieve Atlas Search indexes for a given collection."""      
        try:      
            collection = self.mongo_client[db_name][collection_name]      
            indexes = collection.list_search_indexes()      
            index_info_list = []      
            for index in indexes:      
                index_info = {      
                    'name': index.get('name', ''),      
                    'status': index.get('status', ''),      
                    'embedding_field': ''      
                }      
                definition = index.get('definition', {})      
                mappings = definition.get('mappings', {})      
                fields = mappings.get('fields', {})      
                for field_name, field_info in fields.items():      
                    if field_info.get('type') == 'vector' or field_info.get('type') == 'knnVector':      
                        index_info['embedding_field'] = field_name      
                        break      
                index_info_list.append(index_info)      
            return index_info_list      
        except PyMongoError as e:      
            self.logger.error(f"Error listing search indexes for {db_name}.{collection_name}: {e}")      
            return []      
        except AttributeError:      
            self.logger.error("The 'list_search_indexes' method requires PyMongo 4.8 or higher.")      
            return []      
      
    def create_embedding_search_index(self, db_name, collection_name, index_name, embedding_field, num_dimensions):      
        """Create a new Atlas Search index for the specified embedding field."""      
        try:      
            collection = self.mongo_client[db_name][collection_name]      
      
            existing_indexes = collection.list_search_indexes()      
            for index in existing_indexes:      
                if index.get('name') == index_name:      
                    raise ValueError(f"An index with the name '{index_name}' already exists.")      
      
            search_index_model = SearchIndexModel(      
                definition={      
                    "fields": [      
                        {      
                            "type": "vector",      
                            "path": embedding_field,      
                            "numDimensions": num_dimensions,      
                            "similarity": "cosine"      
                        }      
                    ]      
                },      
                name=index_name,      
                type="vectorSearch"      
            )      
      
            collection.create_search_index(search_index_model)      
      
            self.logger.info(f"Index '{index_name}' creation initiated.")      
            return True      
        except Exception as e:      
            self.logger.error(f"Error creating search index {index_name} for {db_name}.{collection_name}: {e}")      
            return False      
      
    def generate_embeddings_for_collection(      
        self,      
        source_db_name,      
        source_collection_name,      
        source_field,      
        match_stage,      
        embedding_models,      
        records_limit,      
        destination_db_name,      
        destination_collection_name):      
        """Clones documents from source to destination collection, applies match_stage if any, generates embeddings in destination collection."""      
        source_collection = self.mongo_client[source_db_name][source_collection_name]      
        destination_collection = self.mongo_client[destination_db_name][destination_collection_name]      
      
        # Cloning documents with match_stage      
        query = match_stage if match_stage else {}      
        cursor = source_collection.find(query).limit(records_limit)      
      
        self.logger.info(f"Cloning documents from {source_db_name}.{source_collection_name} to {destination_db_name}.{destination_collection_name} with match stage: {match_stage}")      
      
        documents_to_insert = []      
        for doc in cursor:      
            # Remove _id to avoid duplicate key error if copying back into same collection      
            doc['_id'] = ObjectId()  # Create new _id for each doc      
            documents_to_insert.append(doc)      
      
        if documents_to_insert:      
            result = destination_collection.insert_many(documents_to_insert)      
            self.logger.info(f"Inserted {len(result.inserted_ids)} documents into {destination_db_name}.{destination_collection_name}")      
        else:      
            self.logger.info(f"No documents to insert into {destination_db_name}.{destination_collection_name}")      
            return      
      
        # Now, generate embeddings in destination collection      
        documents_to_update = destination_collection.find(      
            {source_field: {'$exists': True, '$ne': None}},      
            {'_id': 1, source_field: 1}      
        ).limit(records_limit)      
      
        total_docs = destination_collection.count_documents(      
            {source_field: {'$exists': True, '$ne': None}}      
        )      
      
        self.logger.info(f"Generating embeddings for up to {records_limit} documents in {destination_db_name}.{destination_collection_name}...")      
      
        batch_size = 100      
        documents = []      
        processed_docs = 0      
        for doc in documents_to_update:      
            documents.append(doc)      
            if len(documents) >= batch_size:      
                self.process_documents(documents, destination_collection, source_field, embedding_models)      
                processed_docs += len(documents)      
                self.logger.info(f"Processed {processed_docs}/{records_limit} documents.")      
                documents = []      
        if documents:      
            self.process_documents(documents, destination_collection, source_field, embedding_models)      
            processed_docs += len(documents)      
            self.logger.info(f"Processed {processed_docs}/{records_limit} documents.")      
      
    def process_documents(self, documents, collection, source_field, embedding_models):      
        bulk_ops = []      
        for doc in documents:      
            text_to_embed = self.get_value_from_field(doc, source_field)      
            if text_to_embed and isinstance(text_to_embed, str):      
                update_fields = {}      
                for model_name, model_info in embedding_models.items():      
                    embedding = self.get_embedding(text_to_embed, model_info)      
                    if embedding:      
                        update_fields[model_info['embedding_field_name_in_doc']] = embedding      
                    else:      
                        self.logger.error(f"Failed to generate embedding for model '{model_name}' and document with _id: {doc['_id']}")      
                if update_fields:      
                    bulk_ops.append(      
                        pymongo.UpdateOne(      
                            {'_id': doc['_id']},      
                            {'$set': update_fields}      
                        )      
                    )      
            else:      
                self.logger.error(f"No valid text to embed for document with _id: {doc['_id']}")      
        if bulk_ops:      
            try:      
                result = collection.bulk_write(bulk_ops)      
                self.logger.info(f"Updated {result.matched_count} documents.")      
            except Exception as e:      
                self.logger.error(f"Error during bulk update: {e}")      
      
    def get_value_from_field(self, document, field_name):      
        """Get the value from the nested field in the document."""      
        fields = field_name.split('.')      
        value = document      
        for field in fields:      
            if isinstance(value, dict) and field in value:      
                value = value[field]      
            else:      
                return None      
        return value      
      
# Create an instance of MDBAutoEval      
autoeval = MDBAutoEval()      
app = autoeval.app      
      
# --- Flask Routes ---      
      
@app.route('/', methods=['GET'])      
def index():      
    # Fetch previous test runs from MongoDB      
    previous_runs = list(      
        autoeval.test_runs_collection.find(      
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
      
    # Fetch idx_meta entries      
    try:      
        idx_meta_entries = list(autoeval.idx_meta_collection.find({}))      
        # Convert ObjectId to string for use in templates      
        for entry in idx_meta_entries:      
            entry['_id'] = str(entry['_id'])      
    except Exception as e:      
        autoeval.logger.error(f"Error fetching idx_meta entries: {e}")      
        idx_meta_entries = []      
      
    # Use the idx_meta_id from session if available      
    if 'idx_meta_id' in session:      
        idx_meta_id = session['idx_meta_id']      
    else:      
        # If no idx_meta_id in session, try to set a default one      
        if idx_meta_entries:      
            idx_meta_id = idx_meta_entries[0]['_id']      
            session['idx_meta_id'] = idx_meta_id      
        else:      
            idx_meta_id = None      
      
    if idx_meta_id:      
        selected_idx_meta = autoeval.idx_meta_collection.find_one({'_id': ObjectId(idx_meta_id)})      
        if selected_idx_meta and 'test_dataset' in selected_idx_meta:      
            test_dataset = selected_idx_meta['test_dataset']      
        else:      
            test_dataset = []      
    else:      
        test_dataset = []      
      
    default_response_criteria = """      
- Provide a concise answer to the question based ONLY on the context.      
- Respond ONLY with the complete title of the movie that best matches the question.      
    """.strip()      
    default_system_prompt = """      
Below is the context, which includes plots of movies.      
      
[context]      
{context}      
[/context]      
    """.strip()      
    default_user_prompt = """      
[response_criteria]      
{response_criteria}      
[/response_criteria]      
      
[question]      
{question}      
[/question]      
    """.strip()      
      
    # Prepare the list of available metrics and mark Factuality as selected by default      
    available_metrics = [{'name': name, 'selected': (name == 'Factuality')} for name in autoeval.METRICS.keys()]      
      
    # Fetch the list of databases for selection      
    databases_info = autoeval.get_databases()      
      
    current_year = datetime.datetime.utcnow().year  # Pass current_year to template      
      
    # Default embedding model name      
    default_embedding_model = autoeval.DEFAULT_EMBEDDING_MODEL_NAME      
      
    return render_template(      
        'index.html',      
        test_dataset=test_dataset,      
        deployment_names=autoeval.DEPLOYMENT_NAMES,      
        default_deployment=autoeval.DEFAULT_DEPLOYMENT_NAME,      
        previous_runs=previous_runs,      
        default_response_criteria=default_response_criteria,      
        default_system_prompt=default_system_prompt,      
        default_user_prompt=default_user_prompt,      
        current_year=current_year,      
        available_metrics=available_metrics,      
        databases=databases_info,  # Use databases_info which includes collections and embedding fields      
        embedding_models=autoeval.EMBEDDING_MODELS,  # Pass embedding models to template      
        default_embedding_model=default_embedding_model,  # Pass default embedding model to template      
        idx_meta_entries=idx_meta_entries  # Pass idx_meta entries to template      
    )      
      
@app.route('/get_fields', methods=['POST'])      
def get_fields_route():      
    idx_meta_id = request.form.get('idx_meta_id')      
    if idx_meta_id:      
        session['idx_meta_id'] = idx_meta_id  # Update the session with the new idx_meta_id      
        try:      
            idx_meta = autoeval.idx_meta_collection.find_one({'_id': ObjectId(idx_meta_id)})      
            if not idx_meta:      
                return jsonify({'fields': []})      
            db_name = idx_meta.get('destination_db')      
            collection_name = idx_meta.get('destination_collection')      
            # Now fetch the fields using these values      
            fields = autoeval.get_collection_fields(db_name, collection_name)      
            # Return fields as a list of dicts with 'name' key      
            fields_info = [{'name': field} for field in fields]      
            return jsonify({'fields': fields_info, 'test_dataset': idx_meta.get('test_dataset', [])})      
        except Exception as e:      
            autoeval.logger.error(f"Error fetching fields by idx_meta_id: {e}")      
            return jsonify({'fields': [], 'test_dataset': []})      
    else:      
        # If idx_meta_id is not provided, handle accordingly      
        return jsonify({'fields': [], 'test_dataset': []})      
      
@app.route('/get_collections', methods=['POST'])      
def get_collections_route():      
    db_name = request.form.get('db_name')      
    collections = autoeval.get_collections(db_name)      
    # Return collections as a list of dicts with 'name' key      
    collections_info = [{'name': coll_name} for coll_name in collections]      
    return jsonify({'collections': collections_info})      
      
@app.route('/get_indexes', methods=['POST'])      
def get_indexes_route():      
    db_name = request.form.get('db_name')      
    collection_name = request.form.get('collection_name')      
    indexes = autoeval.get_atlas_search_indexes(db_name, collection_name)      
    # Return indexes as a list of dicts with 'name' and 'embedding_field' keys      
    index_info_list = [{'name': idx['name'], 'embedding_field': idx.get('embedding_field', '')} for idx in indexes if 'name' in idx]      
    return jsonify({'indexes': index_info_list})      
      
@app.route('/get_vector_field_from_index', methods=['POST'])      
def get_vector_field_from_index_route():      
    db_name = request.form.get('db_name')      
    collection_name = request.form.get('collection_name')      
    index_name = request.form.get('index_name')      
    indexes = autoeval.get_atlas_search_indexes(db_name, collection_name)      
    vector_field = ''      
    for idx in indexes:      
        if idx.get('name') == index_name:      
            vector_field = idx.get('embedding_field', '')      
            break      
    return jsonify({'vector_field': vector_field})      
      
@app.route('/get_vector_fields', methods=['POST'])      
def get_vector_fields_route():      
    db_name = request.form.get('db_name')      
    collection_name = request.form.get('collection_name')      
    embedding_fields = autoeval.get_collection_fields(db_name, collection_name)      
    # Return vector fields as a list of dicts with 'name' key      
    vector_fields_info = [{'name': field} for field in embedding_fields]      
    return jsonify({'vector_fields': vector_fields_info})      
      
@app.route('/add_test_case', methods=['POST'])      
def add_test_case():      
    # Get the test case data from the form      
    input_prompt = request.form.get('new_input_prompt', '').strip()      
    expected_output = request.form.get('new_expected_output', '').strip()      
      
    # Retrieve the idx_meta_id from the session or form      
    idx_meta_id = session.get('idx_meta_id') or request.form.get('idx_meta_id')      
    if not idx_meta_id:      
        return redirect(url_for('index'))      
      
    idx_meta = autoeval.idx_meta_collection.find_one({'_id': ObjectId(idx_meta_id)})      
    if not idx_meta:      
        return redirect(url_for('index'))      
      
    test_dataset = idx_meta.get('test_dataset', [])      
      
    if input_prompt:      
        new_test_case = {      
            'id': len(test_dataset) + 1,      
            'input': input_prompt,      
            'expected': expected_output      
        }      
      
        test_dataset.append(new_test_case)      
        # Update the idx_meta document with the new test_dataset      
        autoeval.idx_meta_collection.update_one(      
            {'_id': ObjectId(idx_meta_id)},      
            {'$set': {'test_dataset': test_dataset}}      
        )      
      
    return redirect(url_for('index'))      
      
@app.route('/remove_test_case', methods=['POST'])      
def remove_test_case():      
    # Get the test case id to remove      
    test_case_id = int(request.form.get('test_case_id_to_remove', '-1'))      
      
    idx_meta_id = session.get('idx_meta_id') or request.form.get('idx_meta_id')      
    if not idx_meta_id:      
        return redirect(url_for('index'))      
      
    idx_meta = autoeval.idx_meta_collection.find_one({'_id': ObjectId(idx_meta_id)})      
    if not idx_meta:      
        return redirect(url_for('index'))      
      
    test_dataset = idx_meta.get('test_dataset', [])      
      
    # Remove the test case with the matching id      
    test_dataset = [tc for tc in test_dataset if tc['id'] != test_case_id]      
      
    # Renumber the ids      
    for idx, tc in enumerate(test_dataset, start=1):      
        tc['id'] = idx      
      
    # Update the idx_meta document with the new test_dataset      
    autoeval.idx_meta_collection.update_one(      
        {'_id': ObjectId(idx_meta_id)},      
        {'$set': {'test_dataset': test_dataset}}      
    )      
      
    return redirect(url_for('index'))      
      
@app.route('/run_test', methods=['POST'])      
def run_test():      
    # Extract test cases from form data      
    test_dataset = autoeval.get_test_dataset_from_form(request.form)      
    # Retrieve the selected deployment names from the form      
    deployment_names = request.form.getlist('deployment_names')      
    if not deployment_names:      
        # Default to the default deployment name if none are selected      
        deployment_names = [autoeval.DEFAULT_DEPLOYMENT_NAME]      
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
    # Get idx_meta_id from the form      
    idx_meta_id = request.form.get('idx_meta_id')      
    if not idx_meta_id:      
        return "Error: Index configuration must be selected.", 400      
    idx_meta = autoeval.idx_meta_collection.find_one({'_id': ObjectId(idx_meta_id)})      
    if not idx_meta:      
        return "Error: Selected index configuration not found.", 400      
      
    # Extract parameters from idx_meta      
    db_name = idx_meta.get('destination_db')      
    collection_name = idx_meta.get('destination_collection')      
    index_name = idx_meta.get('index_name')      
    vector_field = idx_meta.get('embedding_field')      
    embedding_model_name = idx_meta.get('embedding_model_name')      
      
    # Get selected fields      
    selected_fields = request.form.getlist('fields')      
      
    # Check if necessary parameters are provided      
    if not all([db_name, collection_name, index_name, vector_field]):      
        return "Error: Database, collection, index name, and vector field must be specified.", 400      
      
    # If test_dataset is empty, try to load it from idx_meta      
    if not test_dataset:      
        test_dataset = idx_meta.get('test_dataset', [])      
        if not test_dataset:      
            return "Error: No test dataset available.", 400      
      
    # Run the test      
    test_run_data = autoeval.test_rag_task_with_metrics(      
        test_dataset,      
        deployment_names,      
        response_criteria,      
        system_prompt_template,      
        user_prompt_template,      
        selected_metrics,      
        db_name,      
        collection_name,      
        index_name,      
        vector_field,      
        embedding_model_name,      
        selected_fields  # Pass it here      
    )      
      
    # Save test run data to MongoDB      
    inserted_id = autoeval.test_runs_collection.insert_one(test_run_data).inserted_id      
    test_run_data['_id'] = inserted_id  # Include the ID for rendering      
      
    return render_template('test_results.html', test_run=test_run_data, current_year=datetime.datetime.utcnow().year)      
      
      
@app.route('/test_run/<string:run_id>', methods=['GET'])      
def view_test_run(run_id):      
    # Retrieve test run data from MongoDB using the run_id      
    test_run = autoeval.test_runs_collection.find_one({"_id": ObjectId(run_id)})      
    if not test_run:      
        return "Test run not found."      
    return render_template('test_results.html', test_run=test_run, current_year=datetime.datetime.utcnow().year)      
      
@app.route('/reset', methods=['GET'])      
def reset():      
    # Clear the session      
    session.clear()      
    # Redirect to home, defaults will be loaded      
    return redirect(url_for('index'))      
      
@app.route('/list_indexes', methods=['GET', 'POST'])      
def list_indexes():      
    error_message = None      
    success_message = None      
      
    if request.method == 'POST':      
        action = request.form.get('action')      
        if action == 'create_index':      
            # Handle index creation      
            db_name = request.form.get('db_name')      
            collection_name = request.form.get('collection_name')      
            source_field = request.form.get('source_field')      
            embedding_field_prefix = request.form.get('embedding_field')      
            selected_embedding_models = request.form.getlist('embedding_models')      
            records_limit = int(request.form.get('records_limit', '100'))      
            match_stage_str = request.form.get('match_stage', '')      
            if match_stage_str:      
                try:      
                    match_stage = json.loads(match_stage_str)      
                except json.JSONDecodeError as e:      
                    error_message = f"Invalid JSON in match stage: {e}"      
                    match_stage = {}      
            else:      
                match_stage = {}      
      
            # Set destination db and collection      
            destination_db_name = 'mdb_autoevals'      
            destination_collection_name = f"{db_name}__{collection_name}"      
      
            # Ensure that only one source field is specified      
            if not source_field or ',' in source_field or ' ' in source_field.strip():      
                error_message = "Please specify only one source field."      
            else:      
                # Proceed with embedding generation      
                # Prepare embedding models based on selection      
                embedding_models = {}      
                for model_name in selected_embedding_models:      
                    if model_name in autoeval.EMBEDDING_MODELS:      
                        model_info = autoeval.EMBEDDING_MODELS[model_name]      
                        # Generate field names and index names based on model names      
                        model_info = model_info.copy()  # Copy to avoid mutating the original      
                        safe_model_name = autoeval.to_pascal_case(model_name)      
                        # For embedding field name in document      
                        model_info['embedding_field_name_in_doc'] = f"{embedding_field_prefix}{safe_model_name}"      
                        # For index name      
                        model_info['index_name'] = f"{embedding_field_prefix}_{safe_model_name}_search_index"      
                        embedding_models[model_name] = model_info      
                    else:      
                        error_message = f"Model '{model_name}' is not recognized."      
                        break      
      
                if error_message:      
                    pass  # Handle error      
                elif not embedding_models:      
                    error_message = "No valid embedding models selected."      
                else:      
                    try:      
                        # Generate embeddings and clone documents      
                        autoeval.generate_embeddings_for_collection(      
                            db_name, collection_name, source_field, match_stage, embedding_models, records_limit,      
                            destination_db_name, destination_collection_name)      
      
                        # Create indexes for each selected embedding model      
                        for model_name, model_info in embedding_models.items():      
                            embedding_field_name_in_doc = model_info['embedding_field_name_in_doc']      
                            index_name = model_info['index_name']      
                            # Fetch a sample embedding to get dimensions      
                            sample_embedding = autoeval.get_embedding("sample text for dimensions", model_info)      
                            if sample_embedding is None:      
                                raise ValueError(f"Could not get sample embedding for model '{model_name}'.")      
                            num_dimensions = len(sample_embedding)      
                            # Create index      
                            result = autoeval.create_embedding_search_index(      
                                destination_db_name, destination_collection_name, index_name, embedding_field_name_in_doc,      
                                num_dimensions      
                            )      
                            if result:      
                                autoeval.logger.info(f"Index '{index_name}' created successfully for model '{model_name}'.")      
      
                                # Collect metadata for idx_meta collection      
                                idx_meta = {      
                                    "timestamp": datetime.datetime.utcnow(),      
                                    "index_name": index_name,      
                                    "embedding_field": embedding_field_name_in_doc,      
                                    "num_dimensions": num_dimensions,      
                                    "source_db": db_name,      
                                    "source_collection": collection_name,      
                                    "destination_db": destination_db_name,      
                                    "destination_collection": destination_collection_name,      
                                    "match_stage": match_stage,      
                                    "embedding_model_name": model_name,      
                                    "embedding_model_deployment_name": model_info.get('deployment_name') or model_info.get('model_name'),      
                                    "records_limit": records_limit,      
                                    "source_field": source_field,  # Include the source_field in metadata      
                                    # Include the hardcoded test dataset      
                                    "test_dataset": [      
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
                                }      
                                # Insert metadata into idx_meta collection      
                                autoeval.idx_meta_collection.insert_one(idx_meta)      
                            else:      
                                error_message = f"Failed to create index '{index_name}' for model '{model_name}'."      
                                break  # Exit loop on failure      
                        else:      
                            success_message = "Indexes created successfully for selected embedding models."      
                    except ValueError as ve:      
                        error_message = str(ve)      
                    except Exception as e:      
                        error_message = f"An error occurred: {e}"      
      
    # Fetch databases and their collections and indexes      
    databases_info = []      
    mdb_autoevals_db_name = 'mdb_autoevals'      
    try:      
        mdb_autoevals_db = autoeval.mongo_client[mdb_autoevals_db_name]      
        collections = mdb_autoevals_db.list_collection_names()      
        db_info = {      
            'name': mdb_autoevals_db_name,      
            'collections': []      
        }      
        for collection_name in collections:      
            fields = autoeval.get_collection_fields(mdb_autoevals_db_name, collection_name)      
            indexes = autoeval.get_atlas_search_indexes(mdb_autoevals_db_name, collection_name)      
            collection_info = {      
                'name': collection_name,      
                'fields': fields,      
                'indexes': indexes      
            }      
            db_info['collections'].append(collection_info)      
        databases_info.append(db_info)      
    except Exception as e:      
        error_message = f"Error fetching collections: {e}"      
      
    # Now, we also need the list of source databases for the form      
    source_databases_info = autoeval.get_databases()      
      
    current_year = datetime.datetime.utcnow().year      
      
    return render_template(      
        'list_indexes.html',      
        databases=databases_info,      
        source_databases=source_databases_info,      
        current_year=current_year,      
        error_message=error_message,      
        success_message=success_message,      
        embedding_models=autoeval.EMBEDDING_MODELS  # Pass embedding models to template      
    )      
      
@app.route('/manage_evaluators', methods=['GET', 'POST'])      
def manage_evaluators():      
    error_message = None      
    success_message = None      
      
    # Define available evaluator types and subtypes      
    evaluator_types = ['LLMClassifier', 'FunctionEvaluator']      
    function_subtypes = ['regex_match']  # Only 'regex_match' is available      
      
    # Define default LLMClassifier (Aligned with context)      
    default_classifier = {      
        "name": "Sentiment Analyzer",      
        "prompt_template": (      
            "Given the following text:\n\n"      
            "\"{{output}}\"\n\n"      
            "Please rate the sentiment of this text as 'positive', 'neutral', or 'negative'."      
        ),      
        "choice_scores": {      
            "positive": 1.0,      
            "neutral": 0.5,      
            "negative": 0.0      
        },      
        "use_cot": False,      
        "model_deployment_name": autoeval.DEFAULT_DEPLOYMENT_NAME,      
        "temperature": 0.0      
    }      
      
    if request.method == 'POST':      
        action = request.form.get('action')      
        if action == 'create_evaluator':      
            # Handle creation of new evaluator      
            name = request.form.get('name', '').strip()      
            evaluator_type = request.form.get('evaluator_type', '').strip()      
      
            if evaluator_type == 'LLMClassifier':      
                # Collect LLMClassifier parameters      
                prompt_template = request.form.get('prompt_template', '').strip() or default_classifier['prompt_template']      
                choice_scores_str = request.form.get('choice_scores', '').strip()      
                use_cot = request.form.get('use_cot') == 'on'      
                model_deployment_name = request.form.get('model_deployment_name', '').strip() or autoeval.DEFAULT_DEPLOYMENT_NAME      
                temperature = float(request.form.get('temperature', '0.0'))      
      
                # Parse choice_scores_str into a dictionary      
                try:      
                    choice_scores = json.loads(choice_scores_str) if choice_scores_str else default_classifier['choice_scores']      
                except json.JSONDecodeError as e:      
                    error_message = f"Invalid JSON for choice scores: {e}"      
                else:      
                    evaluator_config = {      
                        "name": name,      
                        "type": evaluator_type,      
                        "prompt_template": prompt_template,      
                        "choice_scores": choice_scores,      
                        "use_cot": use_cot,      
                        "model_deployment_name": model_deployment_name,      
                        "temperature": temperature,      
                        "timestamp": datetime.datetime.utcnow(),      
                    }      
      
            elif evaluator_type == 'FunctionEvaluator':      
                # Collect FunctionEvaluator parameters      
                function_subtype = request.form.get('function_subtype', '').strip()      
                parameters_str = request.form.get('parameters', '').strip()      
      
                # Parse parameters_str into a dictionary      
                try:      
                    parameters = json.loads(parameters_str) if parameters_str else {}      
                except json.JSONDecodeError as e:      
                    error_message = f"Invalid JSON for parameters: {e}"      
                else:      
                    evaluator_config = {      
                        "name": name,      
                        "type": evaluator_type,      
                        "function_subtype": function_subtype,      
                        "parameters": parameters,      
                        "timestamp": datetime.datetime.utcnow(),      
                    }      
            else:      
                error_message = "Invalid evaluator type selected."      
      
            if not error_message:      
                # Store or update evaluator metadata      
                evaluator_id = autoeval.store_evaluator_metadata(evaluator_config)      
                if evaluator_id:      
                    # Reload evaluators to update METRICS      
                    autoeval.load_evaluators_from_db()      
                    success_message = f"Evaluator '{name}' stored successfully."      
                else:      
                    error_message = f"Failed to store evaluator '{name}'."      
      
        elif action == 'delete_evaluator':      
            evaluator_id = request.form.get('evaluator_id')      
            if evaluator_id:      
                # Delete the evaluator from the collection      
                result = autoeval.evaluators_collection.delete_one({"_id": ObjectId(evaluator_id)})      
                if result.deleted_count > 0:      
                    # Reload evaluators to update METRICS      
                    autoeval.load_evaluators_from_db()      
                    success_message = "Evaluator deleted successfully."      
                else:      
                    error_message = "Evaluator not found or already deleted."      
            else:      
                error_message = "Evaluator ID is required to delete an evaluator."      
      
    # Fetch evaluators for display      
    evaluators = list(autoeval.evaluators_collection.find({}))      
    for evaluator in evaluators:      
        evaluator['_id'] = str(evaluator['_id'])      
        evaluator_type = evaluator.get('type')      
        if evaluator_type == 'LLMClassifier':      
            evaluator['choice_scores_str'] = json.dumps(evaluator.get('choice_scores', {}), indent=2)      
        elif evaluator_type == 'FunctionEvaluator':      
            evaluator['parameters_str'] = json.dumps(evaluator.get('parameters', {}), indent=2)      
      
    # Fetch deployment names for the model selection dropdown      
    deployment_names = autoeval.DEPLOYMENT_NAMES      
    default_deployment = autoeval.DEFAULT_DEPLOYMENT_NAME      
      
    current_year = datetime.datetime.utcnow().year      
      
    return render_template(      
        'manage_evaluators.html',      
        evaluators=evaluators,      
        error_message=error_message,      
        success_message=success_message,      
        current_year=current_year,      
        deployment_names=deployment_names,      
        default_deployment=default_deployment,      
        evaluator_types=evaluator_types,      
        function_subtypes=function_subtypes,      
        default_classifier=default_classifier  # Pass the default classifier to the template      
    )      
      
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
        deployment_names = [autoeval.DEFAULT_DEPLOYMENT_NAME]      
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
    # Get idx_meta_id from the form      
    idx_meta_id = request.form.get('idx_meta_id')      
    if not idx_meta_id:      
        return "Error: Index configuration must be selected.", 400      
    idx_meta = autoeval.idx_meta_collection.find_one({'_id': ObjectId(idx_meta_id)})      
    if not idx_meta:      
        return "Error: Selected index configuration not found.", 400      
      
    # Extract parameters from idx_meta      
    db_name = idx_meta.get('destination_db')      
    collection_name = idx_meta.get('destination_collection')      
    index_name = idx_meta.get('index_name')      
    vector_field = idx_meta.get('embedding_field')      
    embedding_model_name = idx_meta.get('embedding_model_name')      
      
    # Get selected fields      
    selected_fields = request.form.getlist('fields')      
      
    # Now process each selected deployment      
    deployment_results = []      
      
    for deployment_name in deployment_names:      
        generated_output, messages, context_docs = autoeval.run_rag_task(      
            input_prompt, deployment_name, response_criteria, system_prompt_template, user_prompt_template,      
            db_name, collection_name, index_name, vector_field, embedding_model_name, selected_fields)      
      
        # Reconstruct context_str from context_docs      
        if context_docs:      
            context_str = "\n".join(      
                [json.dumps(doc, default=autoeval.bson_serializer, indent=2) for doc in context_docs]      
            )      
        else:      
            context_str = "No specific context was found."      
      
        metric_results = {}      
      
        for metric_name in selected_metrics:      
            evaluator = autoeval.METRICS.get(metric_name)      
            if not evaluator:      
                autoeval.logger.error(f"Evaluator for {metric_name} is not initialized.")      
                continue      
                  
            print(f"Evaluating {type(evaluator).__name__} for input: {input_prompt}, output: {generated_output}, context: {context_str}")        
                        
            # Prepare inputs for each metric      
            if callable(evaluator) and not hasattr(evaluator, 'eval'):      
                # For function-based metrics like exact_match      
                result = evaluator(      
                    output=generated_output,      
                    expected=expected_output      
                )      
            elif hasattr(evaluator, 'eval'):      
                # For class-based evaluators with an eval method      
                if isinstance(evaluator, Factuality):      
                    print(f"Evaluating Factuality for input: {input_prompt}, output: {generated_output}, expected: {expected_output}")      
                    print(f"Context: {context_str}")      
                    print(f"Evaluator: {evaluator}")      
                    print("expected_output:", expected_output)      
                    # Ensure that 'expected_output' is provided for Factuality evaluation      
                    if not expected_output:      
                        autoeval.logger.error(f"Expected output is required for Factuality evaluation in preview.")      
                        result = type('Result', (object,), {'score': 0, 'reason': 'Expected output is missing.', 'metadata': {}})      
                    else:      
                        result = evaluator.eval(      
                            input=input_prompt,      
                            output=generated_output,      
                            expected=expected_output      
                        )      
                elif isinstance(evaluator, LLMClassifier):      
                    result = evaluator.eval(      
                        output=generated_output      
                    )      
                elif isinstance(evaluator, (ContextRelevancy, Faithfulness)):      
                    result = evaluator.eval(      
                        input=input_prompt,      
                        output=generated_output,      
                        context=context_str      
                    )      
                else:      
                    autoeval.logger.error(f"Evaluator of type {type(evaluator)} for metric {metric_name} is not recognized.")      
                    continue      
            else:      
                autoeval.logger.error(f"Evaluator for {metric_name} is not callable or does not have an eval method.")      
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
      
        # Append result to deployment_results      
        deployment_results.append(test_case_result)      
      
    # Render the template, passing the deployment_results      
    return render_template('preview_content.html', test_case_results=deployment_results)      
      
if __name__ == '__main__':      
    app.run(debug=True)      