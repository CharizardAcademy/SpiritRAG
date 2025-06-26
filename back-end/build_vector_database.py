import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
from torch import float16 # FLash Attention 2 requires float16
from tqdm import tqdm
import torch, gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Allow PyTorch to use expandable segments for CUDA memory allocation


class DocumentIndexer:
    def __init__(self, model_name='Qwen/Qwen3-Embedding-0.6B'):
        self.model = SentenceTransformer(
            model_name, 
            model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "torch_dtype": float16}, # requires accelerate package, pip install accelerate
            tokenizer_kwargs={"padding_side": "left"},)
        self.doc_index = None
        self.doc_texts = []
        self.doc_file_paths = []  # Store the file paths of documents
        self.doc_embeddings = []
        self.sentence_texts = {}
        self.sentence_indexes = {}
        self.d = None  # Embedding dimension

    def assemble_doc_texts(self, path_to_parsed_doc):
        """
        Reads and assembles document text from a parsed JSON file.
        """
        with open(path_to_parsed_doc, 'r', encoding='utf-8') as f:
            doc_json = json.load(f)
        paragraphs = [item['text'] for item in doc_json if item['type'] in ["paragraph", "heading"]]
        return "\n".join(paragraphs)

    def add_documents(self, doc_paths):
        doc_embeddings = []

        for doc_id, path in tqdm(enumerate(doc_paths)):
            doc_text = self.assemble_doc_texts(path)
            self.doc_texts.append(doc_text)
            self.doc_file_paths.append(path)  # Store the file path

            sentences = doc_text.split(". ")  # Simple sentence splitting
            self.sentence_texts[doc_id] = sentences

            sentence_embeddings = self.model.encode(sentences, convert_to_numpy=True, batch_size=8)
            torch.cuda.empty_cache()
            gc.collect()

            doc_embedding = np.mean(sentence_embeddings, axis=0)
            doc_embeddings.append(doc_embedding)

            d = sentence_embeddings.shape[1]
            sentence_index = faiss.IndexFlatIP(d)
            sentence_index.add(sentence_embeddings)
            self.sentence_indexes[doc_id] = sentence_index

        self.d = doc_embeddings[0].shape[0]
        self.doc_index = faiss.IndexFlatIP(self.d)
        doc_embeddings = np.array(doc_embeddings, dtype=np.float16)
        self.doc_index.add(doc_embeddings)

        print(f"Added {len(doc_paths)} documents to FAISS.")

    def save_index(self, index_path="faiss_docs.bin", metadata_path="faiss_docs.npy"):
        """
        Save the FAISS index and metadata (document content and file paths).
        """
        faiss.write_index(self.doc_index, index_path)

        # Combine document content and file paths into metadata
        metadata = [
            {"content": doc_text, "file_path": file_path}
            for doc_text, file_path in zip(self.doc_texts, self.doc_file_paths)
        ]
        np.save(metadata_path, np.array(metadata, dtype=object))
        print(f"FAISS index saved to {index_path}. Metadata saved to {metadata_path}.")

def get_all_json_files(base_dir):
    """
    Recursively find all JSON files in the given directory that end with 'XXX-en-parsed.json'.
    """
    json_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            
            if file.endswith("en-parsed.json"):
                json_files.append(os.path.join(root, file))
    return json_files

if __name__ == "__main__":
    indexer = DocumentIndexer()

    # Base directory containing parsed data
    base_dir = "/home/yingqiang/projects/spirituality/data/health"

    # Get all JSON file paths
    doc_paths = get_all_json_files(base_dir + "/parsed_data/")

    if not doc_paths:
        print("No valid JSON files found. Exiting.")
    else:
        indexer.add_documents(doc_paths)
        indexer.save_index(base_dir + "/faiss_docs_health_qwen.bin", base_dir + "/faiss_docs_health_qwen.npy")