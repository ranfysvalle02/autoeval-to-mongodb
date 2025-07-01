import os  
import logging  
from datetime import datetime  
from dotenv import load_dotenv  
from pymongo import MongoClient  
from pymongo.errors import ConnectionFailure, PyMongoError  
from openai import AzureOpenAI  
from autoevals.llm import LLMClassifier  
  
# --- Configuration and Initialization ---  
  
# Load environment variables from a .env file if it exists  
load_dotenv()  
  
# Configure logging  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
# Configuration variables  
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')  
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')  
MONGO_URI = os.getenv('MONGO_URI')  
OPENAI_AZURE_DEPLOYMENT_NAME = os.getenv('OPENAI_AZURE_DEPLOYMENT_NAME', 'gpt-4o')  
  
# Validate configuration variables  
required_vars = {  
    'AZURE_OPENAI_API_KEY': AZURE_OPENAI_API_KEY,  
    'AZURE_OPENAI_ENDPOINT': AZURE_OPENAI_ENDPOINT,  
    'MONGO_URI': MONGO_URI,  
    'OPENAI_AZURE_DEPLOYMENT_NAME': OPENAI_AZURE_DEPLOYMENT_NAME  
}  
  
missing_vars = [var for var, value in required_vars.items() if not value]  
if missing_vars:  
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")  
  
# Initialize global clients  
mongo_client = None  
openai_client = None  
  
def initialize_clients():  
    """Initialize Azure OpenAI and MongoDB clients."""  
    global openai_client, mongo_client  
    try:  
        # Initialize Azure OpenAI client  
        openai_client = AzureOpenAI(  
            azure_endpoint=AZURE_OPENAI_ENDPOINT,  
            api_key=AZURE_OPENAI_API_KEY,  
            api_version="2024-02-01"  
        )  
        logger.info("Azure OpenAI client initialized successfully.")  
    except Exception as e:  
        logger.exception("Failed to initialize Azure OpenAI client.")  
        raise  
  
    try:  
        # Initialize MongoDB client  
        mongo_client = MongoClient(MONGO_URI)  
        mongo_client.admin.command('ping')  # Test the connection  
        logger.info("MongoDB client connected successfully.")  
    except ConnectionFailure:  
        logger.exception("Could not connect to MongoDB.")  
        raise  
    except PyMongoError as e:  
        logger.exception("An error occurred with MongoDB.")  
        raise  
  
# --- Core Functions ---  
  
def classify_and_evaluate(text, name, prompt_template, choice_scores, model_deployment_name, use_cot=False):  
    """  
    Classifies the given text using LLMClassifier and returns the result.  
    """  
    try:  
        classifier = LLMClassifier(  
            name=name,  
            prompt_template=prompt_template,  
            choice_scores=choice_scores,  
            model=model_deployment_name,  
            temperature=0.0,  
            client=openai_client,  
            use_cot=use_cot  
        )  
        eval_result = classifier(output=text)  
        return {  
            "name": eval_result.name,  
            "score": eval_result.score,  
            "value": eval_result.metadata.get("choice"),  
            "metadata": eval_result.metadata,  
            "error": eval_result.error  
        }  
    except Exception as e:  
        logger.exception(f"Error during classification for '{name}'.")  
        return {"error": str(e)}  
  
def store_evaluation_result(collection_name, data):  
    """  
    Stores the evaluation result in the specified MongoDB collection.  
    """  
    try:  
        db = mongo_client["LLMClassifierEvalsDB"]  
        collection = db[collection_name]  
        result = collection.insert_one(data)  
        logger.info(f"Stored result in '{collection_name}' with ID: {result.inserted_id}")  
        return result.inserted_id  
    except PyMongoError:  
        logger.exception(f"Failed to store result in MongoDB collection '{collection_name}'.")  
        return None  
  
# --- Main Execution ---  
  
def main():  
    try:  
        initialize_clients()  
  
        # Define test cases  
        test_cases = [  
            {  
                "text": "This product is absolutely amazing! I love it.",  
                "name": "Sentiment Analyzer",  
                "prompt_template": (  
                    "Given the following text:\n\n"  
                    "\"{{output}}\"\n\n"  
                    "Please rate the sentiment of this text as 'positive', 'negative', or 'neutral'."  
                ),  
                "choice_scores": {"positive": 1.0, "neutral": 0.5, "negative": 0.0},  
                "collection_name": "sentiment_evals",  
                "use_cot": False  
            },  
            {  
                "text": "Claim your free prize now! Click this suspicious link: example.com/free-money",  
                "name": "Spam Detector",  
                "prompt_template": (  
                    "Given the following email content:\n\n"  
                    "\"{{output}}\"\n\n"  
                    "Is this email 'spam' or 'not spam'? Explain your reasoning first."  
                ),  
                "choice_scores": {"spam": 0.0, "not spam": 1.0},  
                "collection_name": "spam_evals",  
                "use_cot": True  
            },  
            {  
                "text": "My internet is not working. I need technical assistance.",  
                "name": "Query Router",  
                "prompt_template": (  
                    "Given the customer query below:\n\n"  
                    "\"{{output}}\"\n\n"  
                    "Please route it to the correct department: 'Technical Support', 'Billing', or 'General Inquiry'."  
                ),  
                "choice_scores": {"Technical Support": 1.0, "Billing": 0.5, "General Inquiry": 0.2},  
                "collection_name": "query_routing_evals",  
                "use_cot": False  
            }  
        ]  
  
        # Process each test case  
        for test in test_cases:  
            logger.info(f"Processing: {test['name']}")  
            eval_result = classify_and_evaluate(  
                text=test["text"],  
                name=test["name"],  
                prompt_template=test["prompt_template"],  
                choice_scores=test["choice_scores"],  
                model_deployment_name=OPENAI_AZURE_DEPLOYMENT_NAME,  
                use_cot=test["use_cot"]  
            )  
  
            if eval_result and "score" in eval_result:  
                logger.info(f"Classification Result for '{test['name']}': {eval_result['value']}")  
                data_to_store = {  
                    "original_text": test["text"],  
                    "evaluator_name": test["name"],  
                    "autoevals_result": eval_result,  
                    "timestamp": datetime.utcnow()  
                }  
                store_evaluation_result(test["collection_name"], data_to_store)  
            else:  
                logger.error(f"Failed to classify '{test['name']}': {eval_result.get('error')}")  
  
    except Exception as e:  
        logger.exception("An unexpected error occurred during execution.")  
    finally:  
        if mongo_client:  
            mongo_client.close()  
            logger.info("MongoDB client closed.")  
  
if __name__ == "__main__":  
    main()  