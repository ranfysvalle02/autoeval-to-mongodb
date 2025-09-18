# demo.py

import json
import pandas as pd
from mdb_autoevals import MDBAutoEval
import logging

# Configure logging to show info-level messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data_from_csv(file_path: str) -> list:
    """
    Loads test data from a CSV file into the format expected by the evaluator.
    
    The CSV must have 'input' and 'expected' columns.
    Optional columns 'generated_output' and 'context' are also supported.
    """
    try:
        # Use pandas to read the CSV, which handles missing values gracefully
        df = pd.read_csv(file_path)
        
        # Replace pandas's NaN with None for JSON compatibility and easier checks
        df = df.where(pd.notna(df), None)
        
        test_dataset = []
        for index, row in df.iterrows():
            test_case = {
                'id': index + 1,
                'input': row['input'],
                'expected': row['expected']
            }
            # Add optional fields only if they exist in the CSV row
            if 'generated_output' in row and row['generated_output'] is not None:
                test_case['generated_output'] = row['generated_output']
            if 'context' in row and row['context'] is not None:
                test_case['context'] = row['context']
                
            test_dataset.append(test_case)
            
        logger.info(f"Successfully loaded {len(test_dataset)} test cases from {file_path}")
        return test_dataset
        
    except FileNotFoundError:
        logger.error(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        logger.error(f"An error occurred while reading the CSV file: {e}")
        return []

def main():
    """
    Main function to run the RAG evaluation demo.
    """
    logger.info("Starting MDBAutoEval Demo...")
    
    # --------------------------------------------------------------------------
    # 1. INITIALIZATION
    # --------------------------------------------------------------------------
    # Instantiate the evaluator class.
    # This will load environment variables (.env file), and initialize clients
    # for Azure, Voyage, and MongoDB.
    # Make sure your .env file has:
    # AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, VOYAGE_API_KEY, MONGO_URI
    try:
        evaluator = MDBAutoEval()
    except SystemExit as e:
        logger.error(f"Failed to initialize MDBAutoEval. Exiting. Error: {e}")
        return

    # --------------------------------------------------------------------------
    # 2. LOAD TEST DATA
    # --------------------------------------------------------------------------
    # Load the test dataset from our CSV file. This dataset contains some rows
    # with pre-generated outputs and some without.
    csv_file = "sample_test.csv"
    test_dataset = load_test_data_from_csv(csv_file)
    
    if not test_dataset:
        logger.error("No test data loaded. Aborting the test run.")
        return
        
    # --------------------------------------------------------------------------
    # 3. CONFIGURE THE TEST RUN
    # --------------------------------------------------------------------------
    # For test cases that DON'T have a pre-generated output, the RAG task
    # will be executed using the parameters defined below.
    
    # --- RAG Task Parameters ---
    # Database and collection details for retrieval
    db_name = "sample_mflix"
    collection_name = "movies"
    # The Atlas Search index to use for the vector search
    index_name = "vector_index_plot"
    # The field in your documents that contains the vector embeddings
    vector_field = "plot_embedding_ada"
    # The embedding model used to create the vectors and for the user query
    embedding_model_name = "text-embedding-ada-002"
    # Fields to return from the MongoDB search results
    selected_fields = ["title", "plot", "year"]
    
    # --- LLM Generation Parameters ---
    # List of models you want to test
    deployment_names = ["gpt-4o-mini"]
    # The criteria for what makes a good response
    response_criteria = "Provide a direct and concise answer to the user's question."
    # System prompt template (context will be injected here)
    system_prompt_template = (
        "You are an expert Q&A system. Your response must be grounded in the "
        "provided context. Do not use outside knowledge.\n\n"
        "Context:\n{context}"
    )
    # User prompt template (question and criteria will be injected here)
    user_prompt_template = (
        "Based on the provided context, please adhere to the following criteria: "
        "'{response_criteria}'.\n\nQuestion: {question}"
    )
    
    # --- Evaluation Parameters ---
    # Select which metrics to run
    selected_metrics = ['Factuality', 'Faithfulness', 'Context Relevancy', 'Exact Match']
    
    # --------------------------------------------------------------------------
    # 4. EXECUTE THE TEST RUN
    # --------------------------------------------------------------------------
    logger.info("Starting the RAG task evaluation...")
    test_run_results = evaluator.test_rag_task_with_metrics(
        test_dataset=test_dataset,
        deployment_names=deployment_names,
        response_criteria=response_criteria,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_template,
        selected_metrics=selected_metrics,
        db_name=db_name,
        collection_name=collection_name,
        index_name=index_name,
        vector_field=vector_field,
        embedding_model_name=embedding_model_name,
        selected_fields=selected_fields
    )

    # --------------------------------------------------------------------------
    # 5. DISPLAY RESULTS
    # --------------------------------------------------------------------------
    if test_run_results:
        logger.info("Test run completed. Displaying results:")
        # Pretty-print the JSON results
        print(json.dumps(test_run_results, indent=2))
        
        # You can optionally save the results to a file
        # with open("test_run_results.json", "w") as f:
        #     json.dump(test_run_results, f, indent=2)
        # logger.info("Results saved to test_run_results.json")
    else:
        logger.error("The test run did not produce any results.")

if __name__ == "__main__":
    main()

"""
....... Example output (truncated for brevity) .........................................
{
          "test_case_id": 4,
          "input_prompt": "What is the capital of Canada?",
          "expected_output": "The capital of Canada is Ottawa.",
          "generated_output": "The capital of Canada is Ottawa.",
          "metric_results": {
            "Factuality": {
              "score": 1,
              "reason": "",
              "metadata": {
                "choice": "C",
                "rationale": "1. The question asks for the capital of Canada.\n2. The expert answer states that the capital of Canada is Ottawa.\n3. The submitted answer also states that the capital of Canada is Ottawa.\n4. Both answers provide the same factual information regarding the capital of Canada.\n5. There are no additional details or conflicting information in either answer.\n6. Since both answers contain the exact same factual content, the submitted answer contains all the same details as the expert answer.\n\nBased on this analysis, the correct choice is (C) The submitted answer contains all the same details as the expert answer."
              }
            },
            "Faithfulness": {
              "score": 0.0,
              "reason": "",
              "metadata": {
                "statements": [
                  "The capital of Canada is Ottawa."
                ],
                "faithfulness": [
                  {
                    "statement": "The capital of Canada is Ottawa.",
                    "verdict": 0,
                    "reason": "The context states that no specific context was found, making it impossible to verify the statement."
                  }
                ]
              }
            },
            "Context Relevancy": {
              "score": 0.0,
              "reason": "",
              "metadata": {
                "relevant_sentences": []
              }
            },
            "Exact Match": {
              "score": 1,
              "reason": "Exact match",
              "metadata": {
                "expected": "The capital of Canada is Ottawa.",
                "output": "The capital of Canada is Ottawa."
              }
            }
          },
          "messages": [
            {
              "role": "system",
              "content": "You are an expert Q&A system. Your response must be grounded in the provided context. Do not use outside knowledge.\n\nContext:\nNo specific context was found."
            },
            {
              "role": "user",
              "content": "Based on the provided context, please adhere to the following criteria: 'Provide a direct and concise answer to the user's question.'.\n\nQuestion: What is the capital of Canada?"
            }
          ],
          "context_docs": [],
          "deployment_name": "gpt-4o-mini"
        }
      ],
      "duration_seconds": 37.5542505409976
    }
  },
  "total_duration_seconds": 37.55443279200699,
  "db_name": "sample_mflix",
  "collection_name": "movies",
  "index_name": "vector_index_plot",
  "vector_field": "plot_embedding_ada",
  "embedding_model_name": "text-embedding-ada-002",
  "selected_fields": [
    "title",
    "plot",
    "year"
  ]
}
"""