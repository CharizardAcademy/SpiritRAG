from flask import Flask, request, jsonify, send_from_directory
from approx_nn_search import DocumentRetriever
from text_generation import ResponseGenerator
from parse_pdf import DocumentParser
from assemble_prompt import PromptAssembler
from flask_cors import CORS
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["DISABLE_FLASH_ATTN"] = "1"
# os.environ["FLASH_ATTENTION_2_ENABLED"] = "0"

import torch
torch.cuda.empty_cache()
import requests
import json
import datetime
import faiss
import numpy as np

"""
on bazille: run with conda env engage-cuda122
on marge: run with conda env spiritrag
"""

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", r".*\.ngrok-free\.app"])


def clean_text(text):
    """
    Clean the parsed text by removing unnecessary characters, extra whitespace, Markdown structures, and formatting issues.

    Args:
        text (str): The raw parsed text.

    Returns:
        str: The cleaned text.
    """
    import re

    text = re.sub(r"&[a-z]+;", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(r"^#+\s", "", text, flags=re.MULTILINE)

    text = re.sub(r"\[.*?\]\(.*?\)", "", text)

    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    text = re.sub(r"`.*?`", "", text)

    text = re.sub(r"(\*\*|\*|__|_)(.*?)\1", r"\2", text)

    text = re.sub(r"\|.*?\|", "", text)

    text = re.sub(r"\|[-\s]*\|", "", text)

    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    text = text.strip()

    return text

def initialize_response_generator(size_key="tiny2", use_quantization=False):
    """
    Initialize the ResponseGenerator only once to avoid redundant model loading.
    """
    global response_generator
    print("🔍 Checking if ResponseGenerator needs initialization...")
    if response_generator is None:
        print("🚀 Initializing ResponseGenerator...")
        response_generator = ResponseGenerator(
            model_size_key=size_key,  # Model size key (e.g., "tiny2", "small", "medium", etc.)
            force_quantization=use_quantization  # Auto mode; set to True or False to force quantization or FP16
        )
    else:
        print("✅ ResponseGenerator is already initialized.")


@app.route('/api/save-eval', methods=['POST'])
def save_evaluation():
    """
    Endpoint to save evaluation results.
    Expects a JSON payload with:
    - query: The user query
    - answer: Dictionary containing the generated answer and its evaluation
    - docs: List of dictionaries, each containing a document file name and its evaluation
    - time: The timestamp when the evaluation was submitted
    """
    data = request.get_json()

    print("Received data:", data)

    if not all(key in data for key in ['query', 'answer', 'docs', 'time']):
        return jsonify({"error": "Missing required fields in the payload"}), 400

    eval_folder = "/home/yingqiang/projects/spirituality/data/eval/"
    os.makedirs(eval_folder, exist_ok=True)  
    eval_file_path = os.path.join(eval_folder, f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    try:
        with open(eval_file_path, 'w') as f:
            f.write(json.dumps(data) + '\n')  

        return jsonify({"message": "Evaluation results saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save evaluation results: {str(e)}"}), 500

@app.route('/api/search', methods=['POST'])
def search_documents():
    """
    Endpoint to perform document search and reranking.
    Expects a JSON payload with:
    - query: The user query
    - top_n: Number of top documents to return (optional, default=3)
    """
    data = request.get_json()
    query = data.get('query')
    top_n = data.get('top_n', 3)

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Use the global SEARCH_DATASET variable to determine which dataset(s) to search
    if SEARCH_DATASET not in ["health", "education", "both"]:
        return jsonify({"error": "Invalid SEARCH_DATASET configuration. Must be 'health', 'education', or 'both'."}), 500

    datasets_to_search = ["health"] if SEARCH_DATASET == "health" else ["education"] if SEARCH_DATASET == "education" else ["health", "education"]

    retrieved_docs = []
    for dataset in datasets_to_search:
        retriever.load_index_for_dataset(dataset)
        prefetched = retriever.prefetch_documents(query, top_k=10)
        reranked_docs = retriever.rerank_documents(query, prefetched, top_n=top_n)
        retrieved_docs.extend(reranked_docs)
        retriever.clear_index()  # Clear the index after each dataset to free memory

    return jsonify(retrieved_docs)

@app.route('/api/generate', methods=['POST'])
def generate_text():
    """
    Endpoint to generate text based on a query, retrieved documents, and parsed PDF content.
    Expects a multipart/form-data payload with:
    - query: The user query
    - file: The uploaded PDF file (optional)
    """
    if not request.content_type.startswith('multipart/form-data'):
        return jsonify({"error": "Unsupported Media Type. Content-Type must be multipart/form-data"}), 415

    query = request.form.get('query')
    pdf_file = request.files.get('file') if 'file' in request.files else None

    if not query:
        return jsonify({"error": "Query is required"}), 400

    parsed_pdf = ""
    if pdf_file:
        temp_path = os.path.join("/tmp", pdf_file.filename)
        pdf_file.save(temp_path)

        try:
            parsed_content = document_parser.parse_pdf(temp_path, device="cuda" if torch.cuda.is_available() else "cpu")
            parsed_pdf = clean_text(parsed_content)
        except Exception as e:
            return jsonify({"error": f"Failed to parse PDF: {str(e)}"}), 500
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    retrieved_docs = []
    retrieved_doc_names = []

    # Dynamically determine datasets to search based on SEARCH_DATASET
    if SEARCH_DATASET not in ["health", "education", "both"]:
        return jsonify({"error": "Invalid SEARCH_DATASET configuration. Must be 'health', 'education', or 'both'."}), 500

    datasets_to_search = ["health"] if SEARCH_DATASET == "health" else ["education"] if SEARCH_DATASET == "education" else ["health", "education"]

    print(f"Searching in datasets: {datasets_to_search}")

    for dataset in datasets_to_search:
        retriever.load_index_for_dataset(dataset)  
        prefetched = retriever.prefetch_documents(query, top_k=10)
        reranked_docs = retriever.rerank_documents(query, prefetched, top_n=5)
        retrieved_docs.extend([list(doc.values())[0] for doc in reranked_docs])
        retrieved_doc_names.extend([list(doc.keys())[0] for doc in reranked_docs])
        retriever.clear_index()  # Clear the index after each dataset to free memory

    try:
        generated_text = response_generator.generate_response(query=query, retrieved_text=retrieved_docs, parsed_pdf=parsed_pdf)
    except Exception as e:
        return jsonify({"error": f"Failed to generate response: {str(e)}"}), 500

    return jsonify({
        "generated_text": generated_text,
        "source_files": retrieved_doc_names
    })

@app.route('/api/parse', methods=['POST'])
def parse_document():
    """
    Endpoint to parse and clean an uploaded PDF document.
    Expects a file upload with the key 'file' and an optional 'device' parameter.
    If the device parameter is not provided, it will automatically use GPU if available.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    device = request.form.get('device') 

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    temp_path = os.path.join("/tmp", file.filename)
    file.save(temp_path)

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        parsed_content = document_parser.parse_pdf(temp_path, device=device)

        cleaned_content = clean_text(parsed_content)

        return jsonify({"parsed_content": parsed_content, "cleaned_content": cleaned_content})
    except Exception as e:
        return jsonify({"error": f"Failed to parse and clean document: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """
    Endpoint to fetch metadata for a given file path.
    Expects a query parameter 'path' with the full path to the metadata.jsonl file.
    Ensures the path aligns with the SEARCH_DATASET configuration.
    """
    metadata_path = request.args.get('path')
    print(f"Received metadata path: {metadata_path}")

    # if not metadata_path or not os.path.exists(metadata_path):
    #     return jsonify({"error": "Metadata file not found"}), 404

    # Validate that the metadata path aligns with the SEARCH_DATASET configuration
    if SEARCH_DATASET == "health" and "education" in metadata_path:
        return jsonify({"error": "Metadata path does not align with the SEARCH_DATASET configuration"}), 400
    if SEARCH_DATASET == "education" and "health" in metadata_path:
        return jsonify({"error": "Metadata path does not align with the SEARCH_DATASET configuration"}), 400

    print(f"Fetching metadata from: {metadata_path}")

    try:
        metadata_list = []
        with open(metadata_path, 'r') as f:
            for line in f:
                metadata_list.append(json.loads(line.strip()))  # Parse each line as JSON
        
        return jsonify(metadata_list)
    except Exception as e:
        return jsonify({"error": f"Failed to read metadata: {str(e)}"}), 500

# @app.route('/api/pdf/<path:filename>')
# def serve_pdf(filename):
#     """
#     Serve PDF files dynamically based on the SEARCH_DATASET configuration.
#     """
#     if SEARCH_DATASET not in ["health", "education", "both"]:
#         return jsonify({"error": "Invalid SEARCH_DATASET configuration. Must be 'health', 'education', or 'both'."}), 500

#     # Dynamically determine the base directory based on SEARCH_DATASET
#     base_dir = "/home/yingqiang/projects/spirituality/data/"
#     if SEARCH_DATASET == "health":
#         file_path = os.path.join(base_dir, "health", filename)
#     elif SEARCH_DATASET == "education":
#         file_path = os.path.join(base_dir, "education", filename)
#     else:  # SEARCH_DATASET == "both"
#         health_path = os.path.join(base_dir, "health", filename)
#         education_path = os.path.join(base_dir, "education", filename)
#         if os.path.exists(health_path):
#             return send_from_directory(os.path.dirname(health_path), os.path.basename(health_path))
#         elif os.path.exists(education_path):
#             return send_from_directory(os.path.dirname(education_path), os.path.basename(education_path))
#         else:
#             return jsonify({"error": "PDF file not found in either dataset."}), 404

#     # Serve the file from the determined base directory
#     if not os.path.exists(file_path):
#         return jsonify({"error": "PDF file not found."}), 404

#     return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path))

@app.route('/api/pdf/<path:filename>')
def serve_pdf(filename):
    base_dir = "/home/yingqiang/projects/spirituality/data"
    file_path = os.path.join(base_dir, filename)
    print(f"Serving file: {file_path}")
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404 
    return send_from_directory(base_dir, filename)

@app.route('/api/semantic_similarity', methods=['POST'])
def semantic_similarity():
    """
    Endpoint to calculate semantic similarity between a response and a list of subjects.
    Expects a JSON payload with:
    - response: The generated response
    - subjects: A list of subjects
    """
    data = request.get_json()
    response = data.get('response')
    subjects = data.get('subjects')

    if not response or not subjects:
        return jsonify({"error": "Both 'response' and 'subjects' are required"}), 400

    try:
        # Encode the response and subjects using the retriever's model
        with torch.no_grad():
            response_embedding = retriever.model.encode([response], convert_to_numpy=True, device=retriever.device)
            subject_embeddings = retriever.model.encode(subjects, convert_to_numpy=True, device=retriever.device)

        # Validate embeddings
        if response_embedding is None or subject_embeddings is None:
            raise ValueError("Failed to generate embeddings for response or subjects.")
        if np.isnan(response_embedding).any() or np.isnan(subject_embeddings).any():
            raise ValueError("Embeddings contain NaN values.")

        # Normalize embeddings for cosine similarity
        response_embedding = response_embedding.astype(np.float32)
        subject_embeddings = subject_embeddings.astype(np.float32)
        faiss.normalize_L2(response_embedding)
        faiss.normalize_L2(subject_embeddings)

        # Compute cosine similarities
        similarities = np.dot(subject_embeddings, response_embedding.T).flatten()

        # Convert similarity scores to Python float
        results = [{"subject": subject, "similarity": float(similarity)} for subject, similarity in zip(subjects, similarities)]
        return jsonify(results)
    except Exception as e:
        print(f"Error in semantic_similarity API: {str(e)}")
        return jsonify({"error": f"Failed to calculate semantic similarity: {str(e)}"}), 500

@app.route('/api/download', methods=['GET'])
def download():
    """
    Endpoint to combine all JSONL files in the evaluation folder and serve them as a single file.
    """
    eval_folder = "/home/yingqiang/projects/spirituality/data/eval/"
    temp_file_path = os.path.join(eval_folder, "eval.jsonl")  

    try:
        # Check for the existence of real eval files
        eval_files = [f for f in os.listdir(eval_folder) if f.startswith("eval-") and f.endswith(".jsonl")]
        if not eval_files:
            return jsonify({"error": "No evaluation files found in the folder"}), 500

        # Combine all eval files into a single file
        with open(temp_file_path, 'w') as outfile:
            for filename in eval_files:
                filepath = os.path.join(eval_folder, filename)
                with open(filepath, 'r') as infile:
                    for line in infile:
                        outfile.write(line)

        return send_from_directory(eval_folder, "eval.jsonl", as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Failed to create or serve the combined file: {str(e)}"}), 500

@app.route('/api/config/search_dataset', methods=['GET'])
def get_search_dataset():
    """
    Endpoint to expose the current SEARCH_DATASET configuration to the frontend.
    """
    if SEARCH_DATASET not in ["health", "education", "both"]:
        return jsonify({"error": "Invalid SEARCH_DATASET configuration."}), 500

    return jsonify({"search_dataset": SEARCH_DATASET})

if __name__ == '__main__':

    # Global variable to hold the ResponseGenerator instance
    response_generator = None 

    # Initialize the ResponseGenerator once at startup

    # retriever = DocumentRetriever('Qwen/Qwen3-Embedding-0.6B')  # Single retriever instance
    retriever = DocumentRetriever('all-MiniLM-L6-v2')  # Single retriever instance for semantic similarity

    initialize_response_generator(size_key="tiny2", use_quantization=False)  # Initialize ResponseGenerator

    document_parser = DocumentParser()

    prompt_assembler = PromptAssembler(template_path="prompt_template.txt")

    SEARCH_DATASET = "education"  # Change to "education" or "both" as needed 

    # For production, set debug=False
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
