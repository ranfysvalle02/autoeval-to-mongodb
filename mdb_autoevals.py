# mdb_autoevals.py  
  
import os  
import sys  
import json  
import time  
import logging  
import datetime  
import re  
import decimal  
from typing import Optional  
from bson import ObjectId  
import uuid  # For generating unique identifiers  
  
import pymongo  
from pymongo import MongoClient  
from pymongo.errors import PyMongoError, OperationFailure  
from pymongo.operations import SearchIndexModel  
from dotenv import load_dotenv  
from autoevals import Factuality, LLMClassifier, init  
from autoevals.ragas import ContextRelevancy, Faithfulness  
import voyageai  # Ensure you have the VoyageAI client installed  
from openai import AzureOpenAI  # Ensure you have the OpenAI client library installed  
  
class MDBAutoEval:  
    def __init__(self):  
        # Load environment variables  
        load_dotenv()  
  
        # Configure logging  
        logging.basicConfig(level=logging.INFO)  
        self.logger = logging.getLogger(__name__)  
  
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
  
    def generate_qa_from_context(self, context_str: str, max_questions: int = 5):  
        """  
        Sends a context string to Azure OpenAI to generate Q&A pairs based on the provided text,  
        formatted as a JSON array of {"Q": "A"} objects.  
  
        Args:  
            context_str (str): The text from which to generate questions and answers.  
            max_questions (int): Maximum number of Q&A pairs to generate.  
  
        Returns:  
            list: A list of dictionaries, where each dictionary has a "Q" (question)  
                  and an "A" (answer) key, or None if an error occurs.  
        """  
        try:  
            prompt_text = f"""  
        Based on the following context, generate a list of concise Question and Answer pairs.  
        Each pair should be an object in a JSON array, with keys "Q" for the question  
        and "A" for the corresponding answer. Focus on factual information present in the text.  
  
        Context:  
        {context_str}  
  
        Example desired output format under the key `result`:  
        [  
            {{ "Q": "What is the capital of France?", "A": "Paris" }},  
            {{ "Q": "Who painted the Mona Lisa?", "A": "Leonardo da Vinci" }}  
        ]  
  
        Please provide the JSON array of Q&A pairs [MAX={max_questions}] derived from the context:  
        """  
  
            response = self.azure_client.chat.completions.create(  
                model="gpt-4o-mini",  
                messages=[  
                    {"role": "system", "content": "You are a helpful assistant designed to extract information and output JSON."},  
                    {"role": "user", "content": prompt_text}  
                ],  
                response_format={"type": "json_object"},  
                temperature=0.0  
            )  
  
            json_response_str = response.choices[0].message.content.strip()  
            parsed_json = json.loads(json_response_str)  
            parsed_json = parsed_json.get('result', parsed_json)  
  
            # Validate the parsed JSON structure  
            if isinstance(parsed_json, list) and all(isinstance(item, dict) and "Q" in item and "A" in item for item in parsed_json):  
                return parsed_json  
            else:  
                self.logger.warning("OpenAI returned JSON that did not fully match the expected Q/A array structure.")  
                return parsed_json  # Return it anyway for inspection  
  
        except json.JSONDecodeError as e:  
            self.logger.error(f"Error decoding JSON response: {e}")  
            self.logger.error(f"Raw response content: {response.choices[0].message.content if 'response' in locals() else 'N/A'}")  
            return None  
        except Exception as e:  
            self.logger.error(f"An unexpected error occurred during Q&A generation: {e}")  
            return None  
  
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
            self.logger.error("Cannot perform vector search, MongoDB client is not initialized.")  
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
            expected_output = form_data.get(f'expected_output_'f'{idx}', '').strip()  
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
                                output=generated_output,  
                                input=input_prompt,  
                                expected=expected_output,  
                                context=context_str  
                            )  
                        elif isinstance(evaluator, (ContextRelevancy, Faithfulness)):  
                            result = evaluator.eval(  
                                input=input_prompt,  
                                output=generated_output,  
                                expected=expected_output,  
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
        """Create an Atlas Search index for the specified embedding field."""  
        try:  
            collection = self.mongo_client[db_name][collection_name]  
  
            # Create the index  
            new_definition = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "numDimensions": num_dimensions,
                            "path": embedding_field,
                            "similarity": "cosine"
                        }
                    ]
                },
                name=index_name,
                type="vectorSearch"
            )
            collection.create_search_index(model=new_definition)  
            self.logger.info(f"Index '{index_name}' created successfully.")  
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