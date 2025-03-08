---
layout: default
title: Implementation Guide
---

# Implementation Guide

This document provides practical guidance on implementing various quantization techniques for large language models (LLMs) as part of the Teeny project.

## Prerequisites

Before starting the quantization process, ensure that your environment meets the following requirements:

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU acceleration)
- Hugging Face Transformers library
- At least 24GB VRAM for 7B parameter models (during quantization)
- Storage space for model weights (varies by model size and quantization level)

## Environment Setup

We recommend creating a dedicated virtual environment:

```bash
# Create a virtual environment
python -m venv teeny_env
source teeny_env/bin/activate  # On Windows: teeny_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Model Preparation

Before quantization, ensure your model is properly prepared:

1. Download the pre-trained model:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   
   model_id = "Qwen/qwen2.5-instruct-14b"
   model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   ```

2. Prepare calibration data:
   ```python
   # Example calibration data for code generation tasks
   calibration_data = [
       "def factorial(n):",
       "class BinarySearchTree:",
       "def quicksort(arr):",
       # Add more domain-specific examples
   ]
   
   # Tokenize calibration data
   calibration_dataset = [tokenizer(text, return_tensors="pt").input_ids for text in calibration_data]
   ```

## GPTQ Quantization

GPTQ is a post-training quantization technique that uses approximate second-order information to quantize weights:

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Configuration
quantize_config = BaseQuantizeConfig(
    bits=4,                # Quantization precision (4-bit)
    group_size=128,        # Group size for weight quantization
    desc_act=False,        # Whether to quantize activations
    sym=True               # Symmetric quantization
)

# Prepare model for quantization
model = AutoGPTQForCausalLM.from_pretrained(
    model_id,
    quantize_config=quantize_config,
    torch_dtype=torch.float16,
)

# Run quantization
model.quantize(
    calibration_dataset,
    cache_examples_on_gpu=True
)

# Save quantized model
model.save_quantized("./models/qwen2.5-instruct-14b-gptq-4bit")
```

## GGUF Quantization

GGUF quantization is typically performed using the `llama.cpp` conversion tool:

```bash
# Clone llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build the conversion tool
make

# Convert Hugging Face model to GGUF format
python convert.py --outfile models/qwen2.5-instruct-14b.gguf --outtype f16 path_to_hf_model

# Quantize to different precision levels
./quantize models/qwen2.5-instruct-14b.gguf models/qwen2.5-instruct-14b-q4_k_m.gguf q4_k_m
```

## AWQ Quantization

AWQ focuses on protecting important weights by analyzing activations:

```python
from awq import AutoAWQForCausalLM

# Load model for AWQ quantization
model = AutoAWQForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)

# Prepare and run AWQ quantization
model.quantize(
    tokenizer,
    calibration_dataset,
    bits=4,
    group_size=128,
    zero_point=True,  # Whether to use zero-point quantization
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "weight_dtype": "int4",
        "export_legacy": False,
    }
)

# Save quantized model
model.save_pretrained("./models/qwen2.5-instruct-14b-awq-4bit")
```

## Loading and Inference

### GPTQ Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/qwen2.5-instruct-14b")

# Load GPTQ model
model = AutoModelForCausalLM.from_pretrained(
    "./models/qwen2.5-instruct-14b-gptq-4bit",
    device_map="auto",
)

# Create pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate text
response = pipe(
    "Write a function to calculate the Fibonacci sequence",
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
)
```

### GGUF Models

```python
from ctransformers import AutoModelForCausalLM

# Load GGUF model
model = AutoModelForCausalLM.from_pretrained(
    "./models/qwen2.5-instruct-14b-q4_k_m.gguf",
    model_type="qwen",
    gpu_layers=20,  # Number of layers to offload to GPU
)

# Define generation parameters
response = model.generate(
    "Write a function to calculate the Fibonacci sequence",
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
)
```

### AWQ Models

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, pipeline

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/qwen2.5-instruct-14b")
model = AutoAWQForCausalLM.from_pretrained("./models/qwen2.5-instruct-14b-awq-4bit")

# Create pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate text
response = pipe(
    "Write a function to calculate the Fibonacci sequence",
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
)
```

## Memory Optimization Techniques

### Gradient Checkpointing

When fine-tuning quantized models, gradient checkpointing can reduce memory usage:

```python
model.gradient_checkpointing_enable()
```

### Activation Offloading

For inference with limited GPU memory:

```python
# CPU offloading configuration
from transformers import AutoConfig

config = AutoConfig.from_pretrained("./models/qwen2.5-instruct-14b-gptq-4bit")
config.gradient_checkpointing = True
config.use_cache = False

model = AutoModelForCausalLM.from_pretrained(
    "./models/qwen2.5-instruct-14b-gptq-4bit",
    config=config,
    device_map="auto",
    offload_folder="offload",
)
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size during quantization
   - Use gradient accumulation
   - Try a more aggressive quantization bit-width

2. **Accuracy Degradation**:
   - Increase calibration dataset size and diversity
   - Try different quantization configurations
   - Consider mixed precision quantization

3. **Slow Inference**:
   - Ensure proper GPU utilization
   - Check for CPU bottlenecks
   - Optimize prompt and generation parameters

## Best Practices

1. **Calibration Data Selection**:
   - Use domain-specific examples
   - Include diverse scenarios
   - Match the distribution of target usage

2. **Hardware-specific Optimization**:
   - Tune GPU layers for GGUF models
   - Optimize thread count for CPU inference
   - Consider tensor parallelism for multi-GPU setups

3. **Evaluation**:
   - Always compare against unquantized baseline
   - Test on task-specific benchmarks
   - Measure both performance and efficiency metrics

## Additional Resources

- [Detailed Documentation](https://github.com/Thunderk3g/Teeny/wiki)
- [Example Notebooks](https://github.com/Thunderk3g/Teeny/tree/main/notebooks)
- [Community Forum](https://github.com/Thunderk3g/Teeny/discussions)

This implementation guide will be updated as new quantization techniques and optimization strategies are developed and tested. 