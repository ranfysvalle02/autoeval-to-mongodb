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

import mdb_autoevals  # Import the MDBAutoEval class from mdb_autoevals.py

# Create an instance of MDBAutoEval  
autoeval = mdb_autoevals.MDBAutoEval()  

SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key")  
app = Flask(__name__)  
app.secret_key = SECRET_KEY  
  
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
  
                                # Collect context from a sample of documents to use for test dataset generation  
                                # --- Aggregation Magic ---

                                # First, get a sample document to identify all potential fields.
                                # In a production environment, you might have a more robust schema discovery mechanism
                                # or rely on predefined schema knowledge.
                                
                                sample_doc = autoeval.mongo_client[destination_db_name][destination_collection_name].find_one({})
                                all_fields = list(sample_doc.keys()) if sample_doc else []

                                # Build the dynamic $project stage
                                project_stage_fields = {"_id": 0} # Exclude _id by default

                                for field in all_fields:
                                    # If the field is the source_field, we still want to project it as is,
                                    # assuming it's not an array that needs to be removed from the final extraction.
                                    # If source_field could be an array that you want to exclude from the context_list,
                                    # the condition below would handle it.
                                    project_stage_fields[field] = {
                                        "$cond": {
                                            "if": {"$eq": [{"$type": f"${field}"}, "array"]},
                                            "then": "$$REMOVE",  # Exclude if it's an array
                                            "else": f"${field}" # Include otherwise
                                        }
                                    }

                                pipeline = [
                                    {"$project": project_stage_fields},
                                    {"$limit": 5} # Apply limit after projection
                                ]

                                sample_docs_cursor = autoeval.mongo_client[destination_db_name][destination_collection_name].aggregate(pipeline)

                                context_list = []
                                for doc in sample_docs_cursor:
                                    # Use autoeval.get_value_from_field on the *processed* document
                                    text = autoeval.get_value_from_field(doc, source_field)
                                    if text:
                                        context_list.append(text)

                                context_str = "\n".join(context_list)
  
                                # Generate Q&A pairs from the context  
                                test_dataset_qa_pairs = autoeval.generate_qa_from_context(context_str, max_questions=5)  
  
                                if test_dataset_qa_pairs:  
                                    # Transform the Q&A pairs into the expected test dataset format  
                                    test_dataset = []  
                                    for idx, qa_pair in enumerate(test_dataset_qa_pairs):  
                                        test_dataset.append({  
                                            "id": idx + 1,  
                                            "input": qa_pair["Q"],  
                                            "expected": qa_pair["A"]  
                                        })  
                                else:  
                                    test_dataset = []  # Fallback to empty test dataset  
  
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
                                    "source_field": source_field,  
                                    "test_dataset": test_dataset  # Use the dynamically generated test dataset  
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
                            expected=expected_output  ,
                            context=context_str,
                            response_criteria=response_criteria
                        )  
                elif isinstance(evaluator, LLMClassifier):  
                    result = evaluator.eval(  
                        output=generated_output,  
                        input=input_prompt,
                        context=context_str,
                        response_criteria=response_criteria
                    )  
                elif isinstance(evaluator, (ContextRelevancy, Faithfulness)):  
                    result = evaluator.eval(  
                        input=input_prompt,  
                        output=generated_output,  
                        context=context_str,
                        response_criteria=response_criteria
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