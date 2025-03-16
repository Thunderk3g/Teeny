#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quantization implementation for Large Language Models
Supports multiple quantization methods including GPTQ, AWQ, and INT8
"""

import os
import argparse
import torch
import time
import psutil
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

# Conditional imports based on available packages
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    GPTQ_AVAILABLE = True
except ImportError:
    GPTQ_AVAILABLE = False

try:
    import optimum
    from optimum.intel import INCQuantizer
    INTEL_AVAILABLE = True
except ImportError:
    INTEL_AVAILABLE = False

try:
    from awq import AutoAWQForCausalLM
    AWQ_AVAILABLE = True
except ImportError:
    AWQ_AVAILABLE = False

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Help messages
QUANTIZATION_HELP = """
Available quantization methods:
- fp16: Half-precision floating point
- int8: 8-bit integer quantization (requires CUDA)
- int8-cpu: CPU-friendly 8-bit integer quantization (no CUDA required)
- gptq: GPTQ 4-bit quantization
- awq: AWQ 4-bit quantization
- gguf: GGUF format (requires external conversion)
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize LLMs with various methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=QUANTIZATION_HELP
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model identifier from Hugging Face Hub"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized_models",
        help="Directory to save quantized models"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        choices=["fp16", "int8", "int8-cpu", "gptq", "awq", "gguf"],
        default="int8",
        help="Quantization method to use"
    )
    
    parser.add_argument(
        "--bits",
        type=int,
        choices=[2, 3, 4, 8, 16],
        default=8,
        help="Bit precision for quantization (some methods only support specific values)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for quantization"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmarks after quantization"
    )
    
    parser.add_argument(
        "--humaneval",
        action="store_true",
        help="Run HumanEval benchmark (requires additional setup)"
    )
    
    parser.add_argument(
        "--tokenizer_id",
        type=str,
        default=None,
        help="Optional separate tokenizer id (if different from model_id)"
    )
    
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=1000,
        help="Number of samples to use for calibration"
    )
    
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["symmetric", "asymmetric"],
        default="symmetric",
        help="Quantization scheme (symmetric or asymmetric/affine)"
    )
    
    parser.add_argument(
        "--granularity", 
        type=str,
        choices=["per-tensor", "per-channel"],
        default="per-tensor",
        help="Quantization granularity"
    )

    return parser.parse_args()

class ModelQuantizer:
    """Class to handle various quantization methods for LLMs"""
    
    def __init__(self, args):
        self.args = args
        self.model_id = args.model_id
        self.tokenizer_id = args.tokenizer_id or args.model_id
        self.output_dir = args.output_dir
        self.method = args.method
        self.bits = args.bits
        self.device = args.device
        self.scheme = args.scheme
        self.granularity = args.granularity
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check if required packages are available
        self._check_requirements()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
    
    def _check_requirements(self):
        """Check if required packages are available for the selected method"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for all quantization methods")
        
        if self.method == "gptq" and not GPTQ_AVAILABLE:
            raise ImportError("auto-gptq is required for GPTQ quantization")
        
        if self.method == "awq" and not AWQ_AVAILABLE:
            raise ImportError("awq is required for AWQ quantization")
            
        if self.method == "gguf":
            logger.warning("GGUF quantization requires external conversion tools like llama.cpp")
    
    def _get_calibration_dataset(self, num_samples=1000):
        """Create a calibration dataset for quantization"""
        try:
            from datasets import load_dataset
            
            logger.info(f"Loading calibration dataset with {num_samples} samples")
            
            # Try to load C4 dataset
            try:
                dataset = load_dataset("c4", "en", split="train", streaming=True)
                dataset = dataset.take(num_samples)
                texts = [item["text"] for item in dataset]
            except Exception:
                # Fallback to wikitext
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                if len(dataset) > num_samples:
                    dataset = dataset.select(range(num_samples))
                texts = [item["text"] for item in dataset if len(item["text"]) > 100]
            
            # Tokenize texts
            encodings = []
            for text in texts:
                encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                encodings.append(encoded)
            
            return encodings
            
        except ImportError:
            logger.warning("datasets package not available, using dummy calibration data")
            # Create dummy dataset
            dummy_texts = [
                "This is a sample text for calibration.",
                "Machine learning models can be quantized to reduce their size.",
                "Quantization maps floating point values to integers."
            ] * (num_samples // 3 + 1)
            dummy_texts = dummy_texts[:num_samples]
            
            encodings = []
            for text in dummy_texts:
                encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                encodings.append(encoded)
            
            return encodings
    
    def quantize_fp16(self):
        """Load model with float16 precision"""
        logger.info("Loading model with float16 precision")
        
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map=self.device if self.device == "auto" else {"": self.device}
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded and quantized in {load_time:.2f} seconds")
        
        # Save quantized model
        output_path = os.path.join(self.output_dir, f"{os.path.basename(self.model_id)}-fp16")
        model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        return model, output_path
    
    def quantize_int8(self):
        """Quantize model to 8-bit integers using bitsandbytes"""
        logger.info("Quantizing model to 8-bit precision")
        
        try:
            start_time = time.time()
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,  # Threshold for outlier detection
                llm_int8_skip_modules=None,
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            # Load and quantize model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map=self.device if self.device == "auto" else {"": self.device}
            )
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded and quantized in {load_time:.2f} seconds")
            
            # Save quantized model
            output_path = os.path.join(self.output_dir, f"{os.path.basename(self.model_id)}-int8")
            model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            return model, output_path
        
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Error with bitsandbytes int8 quantization: {str(e)}")
            logger.info("Falling back to CPU-friendly int8 quantization method...")
            return self.quantize_int8_cpu_friendly()
    
    def quantize_int8_cpu_friendly(self):
        """CPU-friendly alternative to int8 quantization without requiring CUDA"""
        logger.info("Using CPU-friendly int8 quantization (via transformers optimization)")
        
        # Import torch at function level to ensure it's available
        import torch
        
        start_time = time.time()
        
        # Load the model in FP16 first
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto" if self.device == "auto" else {"": self.device}
        )
        
        # Apply dynamic quantization
        try:
            from transformers.utils.quantization_config import DynamicQuantizationConfig
            quantization_config = DynamicQuantizationConfig(bits=8)
            model = model.quantize(quantization_config)
            logger.info("Successfully applied dynamic quantization")
        except (ImportError, AttributeError):
            # If dynamic quantization is not available, use torch's built-in quantization
            logger.info("Dynamic quantization not available, using PyTorch's quantize_dynamic")
            import torch.quantization
            
            # Note: This is a simplified approach and may not work for all model types
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Successfully applied PyTorch dynamic quantization")
            except Exception as e:
                logger.warning(f"Could not apply PyTorch quantization: {str(e)}")
                logger.info("Loading model with optimized memory usage only")
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded with optimizations in {load_time:.2f} seconds")
        
        # Save the model
        output_path = os.path.join(self.output_dir, f"{os.path.basename(self.model_id)}-int8-cpu")
        model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        return model, output_path
    
    def quantize_gptq(self):
        """Quantize model using GPTQ"""
        if not GPTQ_AVAILABLE:
            raise ImportError("auto-gptq is required for GPTQ quantization")
        
        logger.info(f"Quantizing model with GPTQ ({self.bits}-bit)")
        
        # Prepare calibration data
        calibration_dataset = self._get_calibration_dataset(self.args.dataset_size)
        
        # Configure quantization parameters based on scheme and granularity
        quantize_config = BaseQuantizeConfig(
            bits=self.bits,
            group_size=128,  # -1 means per-tensor, positive value means group size
            desc_act=False,  # Whether to quantize activations
            sym=self.scheme == "symmetric"  # True for symmetric, False for asymmetric
        )
        
        # For per-channel quantization, we set group_size to 1
        if self.granularity == "per-channel":
            quantize_config.group_size = 1
        
        # Load FP16 model first
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map=self.device if self.device == "auto" else {"": self.device}
        )
        
        # Convert input tensors to appropriate device
        examples = []
        for encoding in calibration_dataset[:10]:  # Use a subset for quantization
            input_ids = encoding["input_ids"].to(model.device)
            examples.append({"input_ids": input_ids})
        
        start_time = time.time()
        
        # Quantize model
        quantized_model = AutoGPTQForCausalLM.from_pretrained(
            model, 
            quantize_config=quantize_config,
            examples=examples,
            bits=self.bits,
            model_basename=f"{os.path.basename(self.model_id)}-{self.bits}bit-gptq"
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model quantized in {load_time:.2f} seconds")
        
        # Save quantized model
        output_path = os.path.join(self.output_dir, f"{os.path.basename(self.model_id)}-{self.bits}bit-gptq")
        quantized_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        return quantized_model, output_path
    
    def quantize_awq(self):
        """Quantize model using AWQ"""
        if not AWQ_AVAILABLE:
            raise ImportError("awq is required for AWQ quantization")
        
        logger.info(f"Quantizing model with AWQ ({self.bits}-bit)")
        
        # Prepare calibration data
        calibration_dataset = self._get_calibration_dataset(self.args.dataset_size)
        
        # Extract text samples for AWQ
        text_samples = []
        for encoding in calibration_dataset[:32]:  # AWQ typically needs fewer samples
            text = self.tokenizer.decode(encoding["input_ids"][0])
            text_samples.append(text)
        
        start_time = time.time()
        
        # Quantize model
        quantized_model = AutoAWQForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            device_map=self.device if self.device == "auto" else {"": self.device}
        )
        
        # AWQ quantization
        quantized_model.quantize(
            text_samples,
            batch_size=1,
            bits=self.bits,
            sym=self.scheme == "symmetric",
            group_size=128 if self.granularity == "per-tensor" else -1
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model quantized in {load_time:.2f} seconds")
        
        # Save quantized model
        output_path = os.path.join(self.output_dir, f"{os.path.basename(self.model_id)}-{self.bits}bit-awq")
        quantized_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        return quantized_model, output_path
    
    def quantize(self):
        """Quantize model using the specified method"""
        logger.info(f"Quantizing {self.model_id} with {self.method} ({self.bits}-bit)")
        
        try:
            # Choose quantization method
            if self.method == "fp16":
                model, model_path = self.quantize_fp16()
            elif self.method == "int8":
                model, model_path = self.quantize_int8()
            elif self.method == "int8-cpu":
                model, model_path = self.quantize_int8_cpu_friendly()
            elif self.method == "gptq":
                model, model_path = self.quantize_gptq()
            elif self.method == "awq":
                model, model_path = self.quantize_awq()
            elif self.method == "gguf":
                raise NotImplementedError("GGUF quantization requires external conversion tools")
            else:
                raise ValueError(f"Unsupported quantization method: {self.method}")
        except Exception as e:
            logger.warning(f"Error applying {self.method} quantization: {str(e)}")
            logger.info("Falling back to FP16 precision as a last resort")
            model, model_path = self.quantize_fp16()
        
        return model, model_path
    
    def benchmark(self, model, model_path):
        """Benchmark the quantized model"""
        logger.info(f"Benchmarking quantized model at {model_path}")
        
        # Performance metrics
        metrics = {}
        
        # Measure model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        model_size_gb = model_size_mb / 1024
        param_count = sum(p.numel() for p in model.parameters())
        
        metrics["model_size"] = {
            "size_mb": model_size_mb,
            "size_gb": model_size_gb,
            "param_count": param_count,
            "param_count_billions": param_count / 1e9
        }
        
        logger.info(f"Model size: {model_size_mb:.2f} MB ({model_size_gb:.2f} GB)")
        logger.info(f"Parameters: {param_count/1e9:.2f} billion")
        
        # Measure memory usage
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
        metrics["memory_usage"] = {"memory_gb": memory_gb}
        
        logger.info(f"Memory usage: {memory_gb:.2f} GB")
        
        # Measure inference speed
        prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to find the maximum sum subarray.",
            "Summarize the consequences of the industrial revolution."
        ]
        
        total_tokens = 0
        total_time = 0
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Measure generation time
            start_time = time.time()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7
                )
            end_time = time.time()
            
            # Calculate metrics
            tokens_generated = output.shape[1] - inputs.input_ids.shape[1]
            generation_time = end_time - start_time
            
            total_tokens += tokens_generated
            total_time += generation_time
            
            logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f} seconds")
        
        # Calculate average metrics
        avg_generation_time = total_time / len(prompts)
        avg_tokens_per_second = total_tokens / total_time
        
        metrics["inference_speed"] = {
            "avg_generation_time": avg_generation_time,
            "avg_tokens_per_second": avg_tokens_per_second
        }
        
        logger.info(f"Average generation time: {avg_generation_time:.4f} seconds")
        logger.info(f"Tokens per second: {avg_tokens_per_second:.2f}")
        
        # Save metrics
        metrics_file = os.path.join(model_path, "benchmark_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def run_humaneval_sample(self, model):
        """Run a small sample of HumanEval problems"""
        logger.info("Running HumanEval sample problems")
        
        # Sample HumanEval problems
        problems = [
            {
                "task_id": "HumanEval/0",
                "prompt": "def double(x):\n    \"\"\"\n    Given a value x, return double x.\n    >>> double(2)\n    4\n    >>> double(3)\n    6\n    \"\"\"\n",
                "test": "assert double(2) == 4\nassert double(3) == 6\nassert double(0) == 0\nassert double(-5) == -10",
                "entry_point": "double"
            },
            {
                "task_id": "HumanEval/1",
                "prompt": "def has_close_elements(numbers, threshold):\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
                "test": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True",
                "entry_point": "has_close_elements"
            }
        ]
        
        successful = 0
        results = []
        
        for problem in problems:
            logger.info(f"Problem: {problem['task_id']}")
            
            # Format prompt (use typical instruction format)
            prompt = f"Complete the following Python function:\n\n{problem['prompt']}\nYour solution should be correct and efficient."
            
            # Generate solution
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False
                )
            
            # Extract generated code
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract function definition
            try:
                # Try to find the function implementation
                function_text = problem['prompt']
                
                # Look for completion starting from the function signature
                signature_line = problem['prompt'].strip().split('\n')[0]
                start_idx = generated_text.find(signature_line)
                
                if start_idx >= 0:
                    completion = generated_text[start_idx + len(signature_line):]
                    
                    # Extract indented lines
                    lines = []
                    for line in completion.split('\n'):
                        if line.strip() == "" or line.startswith(' ') or line.startswith('\t'):
                            lines.append(line)
                        else:
                            break
                    
                    function_text = signature_line + '\n'.join(lines)
                
                # Create local namespace and execute function
                local_namespace = {}
                exec(function_text, {}, local_namespace)
                
                # Check if function exists
                entry_point = problem['entry_point']
                if entry_point in local_namespace:
                    # Run tests
                    exec(problem['test'], local_namespace)
                    logger.info("✅ Tests passed")
                    successful += 1
                    passed = True
                else:
                    logger.info(f"❌ Function {entry_point} not found")
                    passed = False
                
            except Exception as e:
                logger.info(f"❌ Error: {str(e)}")
                passed = False
            
            results.append({
                "task_id": problem['task_id'],
                "passed": passed,
                "generated_text": generated_text
            })
        
        # Calculate pass rate
        pass_rate = successful / len(problems) if problems else 0
        logger.info(f"HumanEval sample pass rate: {pass_rate * 100:.2f}% ({successful}/{len(problems)})")
        
        return {
            "pass_rate": pass_rate,
            "successful": successful,
            "total": len(problems),
            "results": results
        }

def main():
    args = parse_args()
    
    try:
        # Initialize quantizer
        quantizer = ModelQuantizer(args)
        
        # Quantize the model
        model, model_path = quantizer.quantize()
        
        # Benchmark if requested
        if args.benchmark:
            benchmark_results = quantizer.benchmark(model, model_path)
            
            # Run HumanEval sample if requested
            if args.humaneval:
                humaneval_results = quantizer.run_humaneval_sample(model)
                benchmark_results["humaneval"] = humaneval_results
                
                # Save updated metrics
                metrics_file = os.path.join(model_path, "benchmark_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"Quantization completed. Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error during quantization: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 