import torch
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os

"""
    Usage:
        moilliet     CUDA_VISIBLE_DEVICES=0,1 python text_generation.py 
        bazille      CUDA_VISIBLE_DEVICES=0 python text_generation.py
"""

def get_vram_gb():
    if not torch.cuda.is_available():
        print("❌ No GPU detected.")
        return 0
    device_id = 0
    return torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)

def estimate_fp16_vram_usage(model_params_billion):
    fp16_weights = model_params_billion * 2  # FP16 = 2 bytes per param
    estimated_total = fp16_weights * 1.8  # add ~80% overhead conservatively
    return estimated_total

def recommend_precision(model_size_b, available_vram_gb):
    # estimated_fp16_vram = estimate_fp16_vram_usage(model_size_b)
    if model_size_b <= 1:
        return "float16"
    elif model_size_b <= 2:
        return "float16" if available_vram_gb >= 7 else "awq"
    elif model_size_b <= 7:
        return "float16" if available_vram_gb >= 24 else "awq"
    elif model_size_b <= 14:
        return "float16" if available_vram_gb >= 48 else "awq"
    else:
        return "awq"
 
# Mapping of models
MODEL_MAPPING = {
    "tiny": {"size_b": 0.5, "fp16": "Qwen/Qwen2.5-0.5B-Instruct", "awq": "Qwen/Qwen2.5-0.5B-Instruct-AWQ"},
    "tiny2": {"size_b": 0.6, "fp16": "Qwen/Qwen3-0.6B", "awq": "Qwen/Qwen2.5-0.5B-Instruct-AWQ"},
    "small": {"size_b": 1.7, "fp16": "Qwen/Qwen2.5-1.5B-Instruct", "awq": "Qwen/Qwen2.5-1.5B-Instruct-AWQ"},
    "small2": {"size_b": 1.7, "fp16": "Qwen/Qwen3-1.7B", "awq": "Qwen/Qwen3-1.7B"},
    "medium": {"size_b": 3, "fp16": "Qwen/Qwen2.5-3B-Instruct", "awq": "Qwen/Qwen2.5-3B-Instruct-AWQ"},
    "medium2": {"size_b": 4, "fp16": "Qwen/Qwen3-4B", "awq": "Qwen/Qwen3-4B-AWQ"},
    "big": {"size_b": 7, "fp16": "Qwen/Qwen2.5-7B-Instruct", "awq": "Qwen/Qwen2.5-7B-Instruct-AWQ"},
    "big2": {"size_b": 8, "fp16": "Qwen/Qwen3-8B", "awq": "Qwen/Qwen3-8B-AWQ"},
    "large": {"size_b": 14, "fp16": "Qwen/Qwen2.5-14B-Instruct", "awq": "Qwen/Qwen2.5-14B-Instruct-AWQ"},
    "large2": {"size_b": 16, "fp16": "Qwen/Qwen3-16B", "awq": "Qwen/Qwen3-16B-AWQ"},
    'xlarge': {"size_b": 32, "fp16": "Qwen/Qwen2.5-32B-Instruct", "awq": "Qwen/Qwen2.5-32B-Instruct-AWQ"},
    'xlarge2': {"size_b": 32, "fp16": "Qwen/Qwen3-32B", "awq": "Qwen/Qwen3-32B-AWQ"},
}

class ResponseGenerator:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print("🆕 Creating a new instance of ResponseGenerator")
            cls._instance = super(ResponseGenerator, cls).__new__(cls)
        else:
            print("♻️ Reusing existing instance of ResponseGenerator")
        return cls._instance

    def __init__(self, model_size_key="medium", force_quantization=None):
        """
        Initializes the response generator.
        User only selects model size key ("small", "medium", "big", "large", "xlarge").
        Precision and model names are auto-mapped.
        """
        if hasattr(self, "initialized") and self.initialized:
            return  # Prevent re-initialization

        if model_size_key not in MODEL_MAPPING:
            raise ValueError(f"❌ Invalid model size key '{model_size_key}'. Choose from {list(MODEL_MAPPING.keys())}.")

        model_info = MODEL_MAPPING[model_size_key]
        model_name = model_info["fp16"]
        quantized_model_name = model_info["awq"]
        model_size_b = model_info["size_b"]

        available_vram_gb = get_vram_gb()
        print(f"💡 Available VRAM: {available_vram_gb:.1f} GB")
        print(f"💡 Model size: {model_size_b}B ({model_size_key})")

        # Always load tokenizer from the fp16 model for compatibility
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


        # Careful when changing the gpu_memory_utilization, it can lead to OOM errors, 0.8 works for Qwen3-Embedding-0.6B and Qwen3-0.6B
        if force_quantization is True:
            print(f"🔗 Forced quantization mode: {quantized_model_name} with AWQ")
            self.llm = LLM(model=quantized_model_name, quantization="awq", dtype="auto", trust_remote_code=True, max_model_len=8192, tensor_parallel_size=1, gpu_memory_utilization=0.8)

        elif force_quantization is False:
            print(f"🔗 Forced FP16 mode: {model_name}")
            self.llm = LLM(model=model_name, dtype="float16", trust_remote_code=True, max_model_len=8192, tensor_parallel_size=1, gpu_memory_utilization=0.8)

        else:
            suggested_mode = recommend_precision(model_size_b, available_vram_gb)
            if suggested_mode == "float16":
                print(f"🔗 Auto-selected FP16 mode: {model_name}")
                self.llm = LLM(model=model_name, dtype="float16", trust_remote_code=True, max_model_len=8192, tensor_parallel_size=1, gpu_memory_utilization=0.8)
            else:
                print(f"🔗 Auto-selected quantized mode: {quantized_model_name} with AWQ")
                self.llm = LLM(model=quantized_model_name, quantization="awq", trust_remote_code=True, dtype="auto", max_model_len=8192, tensor_parallel_size=1, gpu_memory_utilization=0.8)

        self.initialized = True

    def generate_response(self, query, retrieved_text, parsed_pdf=""):
        if not retrieved_text:
            return "Sorry, I couldn't find relevant information in the database."

        # Combine retrieved text and parsed PDF content
        context = '\n\n'.join(retrieved_text) + parsed_pdf

        # Define the maximum token limit for the model
        MAX_MODEL_LENGTH = 8192 

        # Truncate the context if it exceeds the maximum token limit
        if len(context) > MAX_MODEL_LENGTH:
            print(f"⚠️ Context length ({len(context)}) exceeds the model limit ({MAX_MODEL_LENGTH}). Truncating...")
            context = context[:MAX_MODEL_LENGTH]

        # Use chat template for prompt formatting
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Answer the question in details using the context information provided."},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"}
        ]
        

        # max_total_tokens = 8192
        # max_gen_tokens = 1024
        # max_prompt_tokens = max_total_tokens - max_gen_tokens

        # Apply chat template and tokenize
        # prompt_tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=False)

        # if len(prompt_tokens) > max_prompt_tokens:
        #     print(f"⚠️ Prompt too long ({len(prompt_tokens)} tokens), truncating to {max_prompt_tokens}")
        #     prompt_tokens = prompt_tokens[:max_prompt_tokens]

        # # Decode back to string
        # prompt = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)

        sampling_params = SamplingParams(
            temperature=0.6,               
            top_p=0.95,      
            top_k=20,                               
            repetition_penalty=1.05,        
            max_tokens=8192, # change here
        )
        outputs = self.llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()

        response = re.sub(r"(?s).*?</think>", "", response, flags=re.IGNORECASE).strip()

        # Ensure the response ends with a complete sentence
        if not response.endswith(('.', '!', '?')):
            response = response.rsplit('.', 1)[0] + '.'

        return response

if __name__ == "__main__":
    query = "What are the key environmental factors in sustainable procurement?"
    retrieved_text = "Sustainable procurement focuses on reducing environmental harm, including carbon footprint, energy efficiency, and resource circularity."

    # ✅ Just change the key and the code handles everything
    generator = ResponseGenerator(
        model_size_key="medium",  # "small", "medium", "big", "large", "xlarge"
        force_quantization=False  # Auto mode, can also force True or False
    )

    response = generator.generate_response(query, retrieved_text)
    print("\nGenerated Response:\n", response)
