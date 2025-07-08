import json
import os
import glob

def compute_overall_average(jsonl_files):
    """
    Compute the overall average relevance, accuracy, usefulness, temporality, actionability
    (from docs) and congruence, coherence, relevance, creativity, engagement (from answer.eval)
    across all JSONL files.

    Args:
        jsonl_files (list): List of paths to JSONL files.

    Returns:
        tuple: Two dictionaries with average scores for document-level and answer-level metrics.
    """
    # Separate dictionaries for document-level and overall-level metrics
    doc_scores = {
        "relevance": [],
        "accuracy": [],
        "usefulness": [],
        "temporality": [],
        "actionability": []
    }
    answer_scores = {
        "congruence": [],
        "coherence": [],
        "relevance": [],
        "creativity": [],
        "engagement": []
    }

    for file_path in jsonl_files:
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)

                # Process document-level metrics (from docs)
                for doc in data.get("docs", []):
                    eval_scores = doc["eval"]
                    for metric, value in eval_scores.items():
                        if value != 0:  # Only include non-zero scores
                            doc_scores[metric].append(value)

                # Process overall evaluation metrics (from answer.eval)
                overall_eval = data.get("answer", {}).get("eval", {})
                for metric, value in overall_eval.items():
                    if value != 0:  # Only include non-zero scores
                        answer_scores[metric].append(value)

    # Compute averages for document-level metrics
    doc_avg = {
        metric: (sum(values) / len(values)) if len(values) > 0 else 0
        for metric, values in doc_scores.items()
    }

    # Compute averages for overall evaluation metrics
    answer_avg = {
        metric: (sum(values) / len(values)) if len(values) > 0 else 0
        for metric, values in answer_scores.items()
    }

    return doc_avg, answer_avg


# Example usage
if __name__ == "__main__":
    # Define the folder containing the JSONL files
    folder_path = "/path/to/your/data/"
    
    # Get all JSONL files in the folder
    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))

    print(len(jsonl_files), "JSONL files found.")
    
    # Compute overall averages
    doc_avg, answer_avg = compute_overall_average(jsonl_files)
    
    # Print document-level averages
    print("Document-Level Averages:")
    for metric, avg in doc_avg.items():
        print(f"  {metric}: {avg:.2f}")
    
    # Print answer-level averages
    print("Answer-Level Averages:")
    for metric, avg in answer_avg.items():
        print(f"  {metric}: {avg:.2f}")
