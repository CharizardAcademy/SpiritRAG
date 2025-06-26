import argparse
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
import torch



class DocumentRetriever:
    def __init__(self, model_name):
        self.model_name = model_name
        print("🚀 Initializing DocumentRetriever with model:", self.model_name)
        if self.model_name == 'Qwen/Qwen3-Embedding-0.6B':
            self.model = SentenceTransformer(self.model_name, model_kwargs={"torch_dtype": torch.float16, "attn_implementation": "flash_attention_2"})
            self.model.tokenizer.padding_side = 'left'
        elif self.model_name == 'all-MiniLM-L6-v2':
            self.model = SentenceTransformer(self.model_name)
        self.doc_index = None
        self.doc_texts = []  # Stores document content
        self.doc_file_paths = []  # Stores corresponding file paths
        self.sentence_texts = {}  # Stores sentences for each document
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

    def clear_index(self):
        """
        Clear the FAISS index and associated metadata from memory to free up GPU space.
        """
        self.doc_index = None
        self.doc_texts = []
        self.doc_file_paths = []
        self.sentence_texts = {}
        torch.cuda.empty_cache()  # Clear unused GPU memory
        print("🔄 FAISS index and metadata cleared from memory.")

    def load_index_for_dataset(self, dataset, use_fp16=True):
        """
        Load the FAISS index and metadata for a specific dataset.

        Args:
            dataset (str): The dataset name ("education" or "health").
            use_fp16 (bool): Whether to cast the FAISS index to float16 precision.
        """

        encoder_model = 'qwen' if 'Qwen' in self.model_name else 'minilm'
        base_dir = "/home/yingqiang/projects/spirituality/data"
        index_path = f"{base_dir}/{dataset}/faiss_docs_{dataset}_{encoder_model}.bin"
        metadata_path = f"{base_dir}/{dataset}/faiss_docs_{dataset}_{encoder_model}.npy"

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Index or metadata file not found for dataset: {dataset}")

        # Load the FAISS index
        self.doc_index = faiss.read_index(index_path)

        # Optionally cast the index to float16
        if use_fp16 and hasattr(self.doc_index, 'storage'):
            self.doc_index.storage = faiss.cast_storage_to_float16(self.doc_index.storage)
            print(f"FAISS index for {dataset} cast to float16 precision.")

        # Load metadata
        metadata = np.load(metadata_path, allow_pickle=True).tolist()
        self.doc_texts = [item['content'] for item in metadata]
        self.doc_file_paths = [item['file_path'] for item in metadata]

        # Preprocess sentences for reranking
        self.sentence_texts = {
            idx: doc.split('. ') for idx, doc in enumerate(self.doc_texts)
        }

        print(f"FAISS index and metadata loaded for dataset: {dataset}.")

    def prefetch_documents(self, query, top_k=5):
        """
        Perform nearest neighbor search using document embeddings and query embedding.
        Ensure at least top_k valid documents are returned.
        """
        # Encode the query to get its embedding
        with torch.no_grad():
            query_embedding = self.model.encode([query], convert_to_numpy=True, device=self.device, )

        # Convert the query embedding to float32 for FAISS compatibility
        query_embedding = query_embedding.astype(np.float32)

        # Normalize the query embedding for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Perform the nearest neighbor search in the FAISS index
        scores, doc_indices = self.doc_index.search(query_embedding, top_k)
        torch.cuda.empty_cache()  # Clear CUDA cache

        # Retrieve the top-k documents and their scores
        results = []
        for i, idx in enumerate(doc_indices[0]):
            if idx != -1:
                doc_text = self.doc_texts[idx]
                doc_name = os.path.basename(self.doc_file_paths[idx])  # Extract the JSON file name
                results.append((idx, doc_text, scores[0][i], doc_name))

        # If fewer than top_k valid documents are retrieved, fetch additional ones
        if len(results) < top_k:
            print(f"⚠️ Retrieved only {len(results)} documents. Fetching additional valid documents...")
            additional_indices = [idx for idx in doc_indices[0] if idx != -1 and idx not in [r[0] for r in results]]
            for idx in additional_indices:
                if len(results) >= top_k:
                    break
                doc_text = self.doc_texts[idx]
                doc_name = os.path.basename(self.doc_file_paths[idx])
                results.append((idx, doc_text, scores[0][doc_indices[0].tolist().index(idx)], doc_name))

        return results

    def rerank_documents(self, query, prefetched_docs, top_n=3, alpha=0.7):
        """
        Rerank documents based on a combination of maximum similarity and average similarity.
        Ensure at least top_n documents are returned.
        """
        if not prefetched_docs:
            print("⚠️ No documents to rerank. Returning empty list.")
            return []

        with torch.no_grad():
            query_embedding = self.model.encode([query], convert_to_numpy=True, device=self.device)

        # Convert query embedding to float32 for FAISS compatibility
        query_embedding = query_embedding.astype(np.float32)

        reranked = []

        for doc_id, doc_text, _, doc_name in prefetched_docs:
            # Extract sentences for the document
            sentences = self.sentence_texts.get(doc_id, doc_text.split('. '))
            if not sentences:
                print(f"⚠️ No sentences found for document {doc_name}. Skipping.")
                continue

            # Encode sentences in small batches
            with torch.no_grad():
                sent_embeddings = self.model.encode(sentences, convert_to_numpy=True, device=self.device)

            # Convert sentence embeddings to float32 for FAISS compatibility
            sent_embeddings = sent_embeddings.astype(np.float32)

            # Normalize sentence embeddings for cosine similarity
            faiss.normalize_L2(sent_embeddings)

            # Compute similarity scores
            sim_scores = np.dot(sent_embeddings, query_embedding.T).flatten()
            max_sim = np.max(sim_scores)
            avg_sim = np.mean(sim_scores)
            combined_score = alpha * max_sim + (1 - alpha) * avg_sim

            reranked.append((doc_name, doc_text, combined_score))

        reranked.sort(key=lambda x: x[2], reverse=True)

        # Ensure at least top_n documents are returned
        if len(reranked) < top_n:
            print(f"⚠️ Reranked only {len(reranked)} documents. Adding additional documents...")
            additional_docs = prefetched_docs[len(reranked):]
            for doc_id, doc_text, _, doc_name in additional_docs:
                if len(reranked) >= top_n:
                    break
                reranked.append((doc_name, doc_text, 0))  # Assign a default score of 0 for additional docs

        top_docs = [{doc_name: doc_text} for doc_name, doc_text, _ in reranked[:top_n]]
        return top_docs

    def parse_pdf(self, pdf_path, device='cpu'):
        if pdf_path.endswith(".json"):
            with open(pdf_path, "r", encoding="utf-8") as f:
                parsed_json = json.load(f)
            paragraphs = [item["text"] for item in parsed_json if item["type"] in ["paragraph", "heading"]]
            return "\n".join(paragraphs)

        if device == 'cpu':
            accelerator_options = AcceleratorOptions(num_threads=8, device=AcceleratorDevice.CPU)
        else:
            accelerator_options = AcceleratorOptions(num_threads=8, device=AcceleratorDevice.CUDA)

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
                InputFormat.ASCIIDOC,
                InputFormat.MD,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        converted = converter.convert(pdf_path)
        return converted.document.export_to_markdown()

    def combine_query_with_pdf(self, user_query, pdf_text):
        template = (
            "User question: {query}\n\n"
            "Relevant uploaded document content:\n{pdf_content}\n\n"
            "Please consider both to answer precisely."
        )
        return template.format(query=user_query, pdf_content=pdf_text)

    def calculate_relevance(self, query, doc):
        """
        Calculate the relevance score between the query and a document.
        This is a placeholder implementation and should be replaced with a proper scoring mechanism.
        """
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        doc_embedding = self.model.encode(doc[1], convert_to_numpy=True)  # doc[1] contains the document text

        query_embedding = query_embedding.astype(np.float32)
        doc_embedding = doc_embedding.astype(np.float32)

        faiss.normalize_L2(query_embedding)
        faiss.normalize_L2(doc_embedding)
        score = np.dot(query_embedding, doc_embedding)  # Cosine similarity
        return score

    def calculate_subject_similarity(self, response, subjects):
        """
        Calculate semantic similarity between a response and a list of subjects.

        Args:
            response (str): The generated response.
            subjects (list): A list of subjects.

        Returns:
            list: A list of dictionaries with subjects and their similarity scores.
        """
        try:
            # Encode the response and subjects
            with torch.no_grad():
                response_embedding = self.model.encode([response], convert_to_numpy=True, device=self.device )
                subject_embeddings = self.model.encode(subjects, convert_to_numpy=True, device=self.device)

            # Validate embeddings
            if response_embedding is None or subject_embeddings is None:
                raise ValueError("Failed to generate embeddings for response or subjects.")

            response_embedding = response_embedding.astype(np.float32)
            subject_embeddings = subject_embeddings.astype(np.float32)

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(response_embedding)
            faiss.normalize_L2(subject_embeddings)

            # Compute cosine similarities
            similarities = np.dot(subject_embeddings, response_embedding.T).flatten()

            # Convert similarity scores to Python float
            results = [{"subject": subject, "similarity": float(similarity)} for subject, similarity in zip(subjects, similarities)]
            return results
        except Exception as e:
            print(f"Error in calculate_subject_similarity: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform nearest neighbor search on document embeddings.")
    parser.add_argument(
        "--dataset",
        choices=["health", "education"],
        required=True,
        help="Specify which document embeddings to use: 'health' or 'education'."
    )
    args = parser.parse_args()

    # Determine the paths based on the selected dataset
    base_dir = "/srv/liri_storage/data/yingqiang/projects/spirituality"
    index_path = f"{base_dir}/{args.dataset}/faiss_docs_{args.dataset}.bin"
    metadata_path = f"{base_dir}/{args.dataset}/faiss_docs_{args.dataset}.npy"

    retriever = DocumentRetriever()
    retriever.load_index(index_path, metadata_path)

    query = "What is the focus of the second phase of the World Programme for Human Rights Education as decided by the Human Rights Council in Resolution A/HRC/RES/12/4?"
    prefetched = retriever.prefetch_documents(query, top_k=10)

    # Print the document names
    print("Prefetched Document Names:")
    for _, _, _, doc_name in prefetched:
        print(doc_name)

    final_retrieved = retriever.rerank_documents(query, prefetched, top_n=3)
    print("Final Retrieved Documents:")
    print(final_retrieved)


