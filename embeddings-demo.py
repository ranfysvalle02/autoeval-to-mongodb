import os
import time
import logging
import copy # For deep copying documents
from typing import List, Optional, Dict, Any, Union
from openai import AzureOpenAI
import voyageai # Import VoyageAI
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel
from pymongo.errors import OperationFailure

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Load environment variables from a .env file
load_dotenv()

# Azure OpenAI Configuration
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = "2024-02-01" # Or your desired API version. Ensure compatibility.

# VoyageAI Configuration
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")

# MongoDB Configuration - Source (OG) Collection
SOURCE_DB_NAME = os.environ.get("SOURCE_DB_NAME", "sample_mflix")
SOURCE_COLLECTION_NAME = os.environ.get("SOURCE_COLLECTION_NAME", "movies")
SOURCE_TEXT_FIELD = os.environ.get("SOURCE_TEXT_FIELD", "plot") # Field to extract text from OG documents

# MongoDB Configuration - Target (New) Database
MONGO_URI = os.environ.get("MONGO_URI")
TARGET_DB_NAME = os.environ.get("TARGET_DB_NAME", "autoevals_embeddings")

# --- Processing Parameters for Efficiency ---
SAMPLES_TO_PROCESS = 2000 # Number of random documents to grab from the source collection
INSERT_BATCH_SIZE = 500 # Number of documents to insert in a single bulk operation
EMBEDDING_API_DELAY = 0.1 # Delay in seconds between individual embedding calls (e.g., 0.1s for 10 RPS) - applies to all API calls now

# --- CLEVER PART: Dynamically set the TARGET_COLLECTION_NAME using PascalCase and '_dot_' separator ---
def to_pascal_case(text: str) -> str:
    """Converts a snake_case or hyphen-case string to PascalCase."""
    return "".join(word.capitalize() for word in text.replace('-', '_').split('_'))

TARGET_COLLECTION_NAME = f"{to_pascal_case(SOURCE_DB_NAME)}_dot_{to_pascal_case(SOURCE_COLLECTION_NAME)}"
logger.info(f"Dynamically set TARGET_COLLECTION_NAME to: '{TARGET_COLLECTION_NAME}'")

# Define your embedding models and their corresponding Atlas Search index names.
# Each 'embedding_field_name_in_doc' defines where the embedding from THIS specific model
# will be stored within the document in the TARGET collection.
# Added a 'provider' key to distinguish between AzureOpenAI and VoyageAI models.
EMBEDDING_MODELS_CONFIG = {
    # --- Azure OpenAI Models ---
    "azure-text-embedding-3-large": {
        "provider": "azure_openai",
        "deployment_name": "text-embedding-3-large",
        "index_name": "azure_embeddings_3_large_search_index",
        "embedding_field_name_in_doc": "plotEmbeddingAzureLarge" # PascalCase for embedding field names too
    },
    "azure-text-embedding-3-small": {
        "provider": "azure_openai",
        "deployment_name": "text-embedding-3-small",
        "index_name": "azure_embeddings_3_small_search_index",
        "embedding_field_name_in_doc": "plotEmbeddingAzureSmall" # PascalCase for embedding field names too
    },
    "azure-text-embedding-ada-002": {
        "provider": "azure_openai",
        "deployment_name": "text-embedding-ada-002",
        "index_name": "azure_embeddings_ada_002_search_index",
        "embedding_field_name_in_doc": "plotEmbeddingAzureAda" # PascalCase for embedding field names too
    },
    # --- VoyageAI Models ---
    "voyage-3-large": { # New model added
        "provider": "voyage_ai",
        "deployment_name": "voyage-3-large",
        "index_name": "voyage_3_large_search_index",
        "embedding_field_name_in_doc": "plotEmbeddingVoyage3Large"
    },
    "voyage-3.5": { # New model added
        "provider": "voyage_ai",
        "deployment_name": "voyage-3.5",
        "index_name": "voyage_3_5_search_index",
        "embedding_field_name_in_doc": "plotEmbeddingVoyage3_5"
    },
    "voyage-3.5-lite": { # New model added
        "provider": "voyage_ai",
        "deployment_name": "voyage-3.5-lite",
        "index_name": "voyage_3_5_lite_search_index",
        "embedding_field_name_in_doc": "plotEmbeddingVoyage3_5Lite"
    }
}

# --- Client Initialization ---
azure_openai_client: Optional[AzureOpenAI] = None
voyage_client: Optional[voyageai.Client] = None

# Initialize Azure OpenAI client
try:
    if not AZURE_ENDPOINT or not AZURE_API_KEY:
        logger.warning("Azure OpenAI endpoint or API key not configured. Azure OpenAI models will be skipped.")
    else:
        azure_openai_client = AzureOpenAI(
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
        )
        logger.info("Successfully initialized AzureOpenAI client.")
except Exception as e:
    logger.error(f"Error initializing AzureOpenAI client: {e}")

# Initialize VoyageAI client
try:
    if not VOYAGE_API_KEY:
        logger.warning("VoyageAI API key not configured. VoyageAI models will be skipped.")
    else:
        voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
        logger.info("Successfully initialized VoyageAI client.")
except Exception as e:
    logger.error(f"Error initializing VoyageAI client: {e}")

# --- Unified Embedding Function ---
def get_embedding_unified(
    text: str,
    model_config: Dict[str, Any],
    input_type: str = "document" # 'document' for indexing, 'query' for search
) -> Optional[List[float]]:
    """
    Generates a vector embedding for a given text using a specified model and provider.
    """
    provider = model_config.get("provider")
    deployment_name = model_config.get("deployment_name")

    if provider == "azure_openai":
        if not azure_openai_client:
            logger.error(f"Azure OpenAI client not initialized. Cannot generate embedding for '{deployment_name}'.")
            return None
        try:
            response = azure_openai_client.embeddings.create(input=[text], model=deployment_name)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding with Azure OpenAI model '{deployment_name}': {e}")
            return None
    elif provider == "voyage_ai":
        if not voyage_client:
            logger.error(f"VoyageAI client not initialized. Cannot generate embedding for '{deployment_name}'.")
            return None
        try:
            # VoyageAI expects input_type for optimizing embeddings for retrieval
            response = voyage_client.embed([text], model=deployment_name, input_type=input_type)
            return response.embeddings[0]
        except voyageai.exceptions.AuthenticationError as e:
            logger.error(f"VoyageAI Authentication Error with model '{deployment_name}': {e}. Check VOYAGE_API_KEY.")
            return None
        except Exception as e:
            logger.error(f"Error generating embedding with VoyageAI model '{deployment_name}': {e}")
            return None
    else:
        logger.error(f"Unknown provider '{provider}' in model configuration.")
        return None

# --- MongoDB Index Management Functions ---
# (These remain the same, as they are generic and accept database/collection names as args)

def create_collection_if_not_exists(client: MongoClient, database_name: str, collection_name: str) -> Collection:
    """Ensures a MongoDB collection exists. Creates it if it doesn't."""
    database = client[database_name]
    collection_names = database.list_collection_names()
    if collection_name not in collection_names:
        logger.info(f"Collection '{collection_name}' does not exist. Creating it now.")
        collection = database[collection_name]
        try:
            # Insert and delete a placeholder to explicitly create the collection
            # MongoDB creates collections implicitly on first insert, but this ensures it's there
            collection.insert_one({"_id": 0, "placeholder": True})
            collection.delete_one({"_id": 0})
            logger.info(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            logger.error(f"Error creating placeholder document in collection '{collection_name}': {e}")
            # Depending on error, you might want to re-raise or exit
    else:
        collection = database[collection_name]
        logger.info(f"Collection '{collection_name}' already exists.")
    return collection

def check_index_exists(client: MongoClient, database_name: str, collection_name: str, index_name: str) -> bool:
    """Checks if a specific Atlas Search index exists."""
    try:
        collection = client[database_name][collection_name]
        indexes = list(collection.list_search_indexes())
        for index in indexes:
            if index.get("name") == index_name:
                logger.info(f"Found existing Atlas Search index '{index_name}'.")
                return True
        logger.info(f"Atlas Search index '{index_name}' does not exist.")
        return False
    except OperationFailure as e:
        logger.error(f"Operation failure checking index '{index_name}' existence: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking index '{index_name}' existence: {e}")
        return False

def check_index_ready_status(client: MongoClient, database_name: str, collection_name: str, index_name: str) -> bool:
    """Checks if the specified Atlas Search index status is 'READY'."""
    try:
        collection = client[database_name][collection_name]
        indexes = list(collection.list_search_indexes())
        for index in indexes:
            if index.get("name") == index_name:
                status = index.get("status", "").upper()
                if status == "READY":
                    return True
                else:
                    logger.info(f"Atlas Search index '{index_name}' current status: {status}")
                    return False
        logger.warning(f"Atlas Search index '{index_name}' not found.")
        return False
    except OperationFailure as e:
        logger.error(f"Operation failure checking index '{index_name}' status: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking index '{index_name}' status: {e}")
        return False

def create_atlas_search_index_if_not_exists(
    mongo_client_inst: MongoClient,
    database_name: str,
    collection_name: str,
    index_name: str,
    model_config: Dict[str, Any], # Now takes the full model config
    distance_metric: str = "cosine"
) -> None:
    """
    Creates an Atlas Search index on the specified collection if it does not already exist.
    It uses the provided embedding_model_deployment_name to determine dimensions.
    """
    try:
        collection = create_collection_if_not_exists(mongo_client_inst, database_name, collection_name)

        if check_index_exists(mongo_client_inst, database_name, collection_name, index_name):
            return

        embedding_field = model_config["embedding_field_name_in_doc"]
        deployment_name = model_config["deployment_name"]

        logger.info(f"Creating Atlas Search index '{index_name}' for collection '{collection_name}' "
                    f"using model '{deployment_name}' (provider: {model_config['provider']}).")

        # Generate a sample embedding to determine the number of dimensions
        sample_embedding = get_embedding_unified("sample text for dimensions", model_config, input_type="document")
        if sample_embedding is None:
            raise ValueError(f"Could not get sample embedding for model '{deployment_name}'. "
                             "Cannot determine dimensions for index. Ensure the API key is valid for this provider.")
        num_dimensions = len(sample_embedding)
        logger.info(f"Detected {num_dimensions} dimensions for model '{deployment_name}'.")

        search_index_model = SearchIndexModel(
            definition={
                "mappings": {
                    "dynamic": False, # Set to True if you want other fields to be indexed automatically
                    "fields": {
                        str(embedding_field): { # Use the passed embedding_field for THIS index
                            "type": "knnVector",
                            "dimensions": num_dimensions,
                            "similarity": distance_metric,
                        }
                    },
                }
            },
            name=index_name,
        )

        collection.create_search_index(model=search_index_model)
        logger.info(f"Atlas Search index '{index_name}' creation initiated successfully.")

    except OperationFailure as e:
        logger.error(f"Operation failed while creating Atlas Search index '{index_name}': {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to create Atlas Search index '{index_name}': {e}")
        raise

def wait_for_index_ready(
    mongo_client_inst: MongoClient,
    database_name: str,
    collection_name: str,
    index_name: str,
    max_attempts: int = 20,
    wait_seconds: int = 5,
) -> bool:
    """Waits until the specified Atlas Search index status is 'READY'."""
    attempt = 0
    while attempt < max_attempts:
        if check_index_ready_status(mongo_client_inst, database_name, collection_name, index_name):
            logger.info(f"Atlas Search index '{index_name}' is READY.")
            return True
        attempt += 1
        logger.info(
            f"Attempt {attempt}/{max_attempts}: Atlas Search index '{index_name}' not READY yet. "
            f"Waiting {wait_seconds} second(s)..."
        )
        time.sleep(wait_seconds)
    logger.error(f"Atlas Search index '{index_name}' did not reach READY status after {max_attempts} attempts.")
    return False

# --- Main Execution ---
if __name__ == "__main__":
    if not MONGO_URI or not SOURCE_DB_NAME or not SOURCE_COLLECTION_NAME or not SOURCE_TEXT_FIELD or not TARGET_DB_NAME:
        logger.error("Missing required MongoDB source/target configuration in .env. Exiting.")
        exit(1)

    # Check if at least one embedding client is initialized
    if not azure_openai_client and not voyage_client:
        logger.critical("Neither Azure OpenAI nor VoyageAI client could be initialized. No embedding models will work. Exiting.")
        exit(1)

    mongo_client: Optional[MongoClient] = None
    try:
        mongo_client = MongoClient(MONGO_URI)
        mongo_client.admin.command('ping') # Test connection
        logger.info(f"Successfully connected to MongoDB Atlas.")

        # --- Step 1: Ensure all necessary Atlas Search Indexes exist on the TARGET collection ---
        logger.info(f"\n--- Step 1: Managing Atlas Search Indexes on TARGET collection: {TARGET_DB_NAME}.{TARGET_COLLECTION_NAME} ---")
        for model_key, config in EMBEDDING_MODELS_CONFIG.items():
            # Check if the required client for this model's provider is initialized
            if config["provider"] == "azure_openai" and not azure_openai_client:
                logger.warning(f"Skipping index creation for '{model_key}' (Azure OpenAI) as client is not initialized.")
                continue
            if config["provider"] == "voyage_ai" and not voyage_client:
                logger.warning(f"Skipping index creation for '{model_key}' (VoyageAI) as client is not initialized.")
                continue

            index_name = config["index_name"]

            logger.info(f"\nProcessing index for model: '{model_key}' (Deployment: '{config['deployment_name']}', Provider: '{config['provider']}')")
            try:
                create_atlas_search_index_if_not_exists(
                    mongo_client_inst=mongo_client,
                    database_name=TARGET_DB_NAME,    # Operate on TARGET database
                    collection_name=TARGET_COLLECTION_NAME, # Operate on TARGET collection
                    index_name=index_name,
                    model_config=config # Pass the full model config
                )
                wait_for_index_ready(
                    mongo_client_inst=mongo_client,
                    database_name=TARGET_DB_NAME,    # Operate on TARGET database
                    collection_name=TARGET_COLLECTION_NAME, # Operate on TARGET collection
                    index_name=index_name
                )
            except Exception as e:
                logger.error(f"Critical error managing index '{index_name}': {e}. Skipping this index.")
                # Decide if you want to exit or continue if an index fails to create

        logger.info("\n--- Step 1: Atlas Search Index Management Complete ---")

        # --- Step 2: Iterate through OG collection, embed, clone, and store in new collection ---
        logger.info(f"\n--- Step 2: Processing documents from OG collection: {SOURCE_DB_NAME}.{SOURCE_COLLECTION_NAME} ---")
        logger.info(f"Target collection for enriched documents: {TARGET_DB_NAME}.{TARGET_COLLECTION_NAME}")

        source_collection = mongo_client[SOURCE_DB_NAME][SOURCE_COLLECTION_NAME]
        target_collection = mongo_client[TARGET_DB_NAME][TARGET_COLLECTION_NAME]

        # Optional: Clear target collection before starting if you want a fresh run
        # BE CAREFUL WITH THIS IN PRODUCTION!
        # target_collection.delete_many({})
        # logger.info(f"Cleared all existing documents from {TARGET_DB_NAME}.{TARGET_COLLECTION_NAME}.")

        total_processed_docs = 0
        total_inserted_docs = 0
        
        # Define the aggregation pipeline for random sampling
        sample_pipeline = [
            {"$match": {SOURCE_TEXT_FIELD: {"$exists": True, "$ne": None, "$ne": ""}}}, # Filter for docs with content
            {"$sample": {"size": SAMPLES_TO_PROCESS}} # Grab a random sample
        ]
        
        logger.info(f"Attempting to grab a random sample of {SAMPLES_TO_PROCESS} documents from '{SOURCE_COLLECTION_NAME}'...")
        cursor = source_collection.aggregate(sample_pipeline)

        insert_batch_list = []
        
        for i, source_doc in enumerate(cursor):
            logger.info(f"\nProcessing document {i+1} of {SAMPLES_TO_PROCESS}: Original ID: {source_doc.get('_id', 'N/A')}")

            # 1. Clone the document
            cloned_doc = copy.deepcopy(source_doc)

            # 2. Handle the _id: remove original and add a reference to it
            original_id = cloned_doc.pop('_id', None)
            if original_id:
                cloned_doc['original_doc_id'] = original_id
            
            # 3. Extract text to embed from the source document's field
            text_to_embed = cloned_doc.get(SOURCE_TEXT_FIELD)
            if not text_to_embed or not isinstance(text_to_embed, str):
                logger.warning(f"  Skipping document {original_id}: No valid text found in '{SOURCE_TEXT_FIELD}' field.")
                total_processed_docs += 1
                continue # Skip this document

            # 4. Generate and add embeddings from all configured models
            embeddings_generated_count = 0
            for model_key, config in EMBEDDING_MODELS_CONFIG.items():
                # Check if the required client for this model's provider is initialized
                if config["provider"] == "azure_openai" and not azure_openai_client:
                    logger.debug(f"  Skipping embedding for '{model_key}' (Azure OpenAI) as client is not initialized.")
                    continue
                if config["provider"] == "voyage_ai" and not voyage_client:
                    logger.debug(f"  Skipping embedding for '{model_key}' (VoyageAI) as client is not initialized.")
                    continue

                embedding_field_name_in_doc = config["embedding_field_name_in_doc"]

                logger.debug(f"  - Embedding with '{model_key}' (Provider: {config['provider']}) for field '{SOURCE_TEXT_FIELD}'...")
                embedding = get_embedding_unified(text_to_embed, config, input_type="document")

                if embedding:
                    cloned_doc[embedding_field_name_in_doc] = embedding
                    embeddings_generated_count += 1
                    logger.debug(f"  - Successfully added embedding from '{model_key}'.")
                else:
                    logger.warning(f"  - Failed to generate embedding for '{model_key}'.")

                # Apply API rate limit delay after each embedding call
                time.sleep(EMBEDDING_API_DELAY)

            if embeddings_generated_count > 0:
                # Add to batch for bulk insert
                insert_batch_list.append(cloned_doc)
                # total_inserted_docs will be incremented after successful bulk insert, or here if we track pending.
                # For simplicity, let's count docs added to batch.
                total_inserted_docs += 1 

                # If batch size is reached, perform bulk insert
                if len(insert_batch_list) >= INSERT_BATCH_SIZE:
                    try:
                        target_collection.insert_many(insert_batch_list)
                        logger.info(f"  Inserted batch of {len(insert_batch_list)} documents.")
                        insert_batch_list = [] # Reset batch
                    except Exception as e:
                        logger.error(f"  Error during bulk insert: {e}")
                        # Optionally handle individual document failures here if needed
                        insert_batch_list = [] # Clear batch to avoid re-attempting failed docs
                        # Consider decrementing total_inserted_docs here if you want accurate count of *successfully* inserted docs
            else:
                logger.warning(f"  No embeddings generated for document (original ID: {original_id}). Skipping insertion.")
            total_processed_docs += 1

        # Insert any remaining documents in the batch after the loop finishes
        if insert_batch_list:
            try:
                target_collection.insert_many(insert_batch_list)
                logger.info(f"  Inserted final batch of {len(insert_batch_list)} documents.")
            except Exception as e:
                logger.error(f"  Error during final bulk insert: {e}")

        logger.info(f"\n--- Step 2: Document Processing Complete. Processed {total_processed_docs} documents, inserted {total_inserted_docs} into {TARGET_COLLECTION_NAME}. ---")

        # --- Step 3: Demonstrate Vector Search on the NEW collection (optional, for verification) ---
        logger.info(f"\n--- Step 3: Demonstrating Vector Search on TARGET collection: {TARGET_DB_NAME}.{TARGET_COLLECTION_NAME} ---")

        search_query_text = "movie about a young boy's adventure" # Example query

        for model_key, config in EMBEDDING_MODELS_CONFIG.items():
            # Check if the required client for this model's provider is initialized
            if config["provider"] == "azure_openai" and not azure_openai_client:
                logger.warning(f"Skipping search for '{model_key}' (Azure OpenAI) as client is not initialized.")
                continue
            if config["provider"] == "voyage_ai" and not voyage_client:
                logger.warning(f"Skipping search for '{model_key}' (VoyageAI) as client is not initialized.")
                continue

            index_name = config["index_name"]
            # The path to the embedding field in the document for THIS specific index
            embedding_field_for_search = config["embedding_field_name_in_doc"]

            logger.info(f"\nPerforming search using index: '{index_name}' (via model '{model_key}', provider: '{config['provider']}')")
            # Generate query embedding using the unified function
            query_vector = get_embedding_unified(search_query_text, config, input_type="query")

            if query_vector:
                try:
                    pipeline = [
                        {
                            "$vectorSearch": {
                                "index": index_name,
                                "queryVector": query_vector,
                                "path": embedding_field_for_search, # Use the specific field name for search
                                "numCandidates": 50,  # Number of candidates to consider
                                "limit": 3,           # Number of results to return
                            }
                        },
                        {"$project": {
                            "_id": 0,
                            "original_doc_id": 1,
                            "title": 1,
                            SOURCE_TEXT_FIELD: 1, # Show the original text field
                            "score": {"$meta": "vectorSearchScore"}
                        }},
                    ]
                    results = list(target_collection.aggregate(pipeline)) # Search the TARGET collection

                    if results:
                        logger.info(f"  Found {len(results)} results:")
                        for i, doc in enumerate(results):
                            print(f"    {i+1}. Title: {doc.get('title', 'N/A')} (Score: {doc.get('score', 0):.4f})")
                            plot_snippet = doc.get(SOURCE_TEXT_FIELD, 'N/A')
                            print(f"       Original Plot: {plot_snippet[:100]}{'...' if len(plot_snippet) > 100 else ''}")
                    else:
                        logger.info("  No results found for this query and index.")
                except Exception as e:
                    logger.error(f"Error during vector search with index '{index_name}': {e}")
            else:
                logger.warning(f"Could not generate query embedding for search using model '{model_key}'. Skipping search for this model.")

        logger.info("\n--- Step 3: Vector Search Demonstration Complete ---")


    except Exception as e:
        logger.critical(f"An unhandled error occurred during MongoDB operations: {e}", exc_info=True)
    finally:
        if mongo_client:
            mongo_client.close()
        logger.info("MongoDB client closed.")