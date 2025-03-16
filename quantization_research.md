# LLM Quantization Research

## Introduction

This document provides an organized overview of quantization techniques for Large Language Models (LLMs). Quantization is a model compression technique that reduces the precision of model weights and activations, significantly decreasing model size and computational requirements while attempting to maintain model performance.

## Quantization Methods Comparison

### GPTQ
GPTQ (Generative Pre-trained Transformer Quantization) is a one-shot weight quantization method specifically designed for transformer models.

- **Key Features**:
  - Uses second-order information (Hessian matrix) to determine optimal quantization
  - Applies quantization layer-by-layer in a single pass
  - Supports extremely low bit-widths (2-4 bits)
  - Minimal accuracy loss compared to other methods
  
- **Implementation**: Uses the `auto-gptq` library
- **Best Use Case**: When minimum model size is critical

### GGUF
GGUF (GPT-Generated Unified Format) is a file format for quantized models that evolved from GGML.

- **Key Features**:
  - Designed for inference efficiency
  - Compatible with llama.cpp ecosystem
  - Various quantization levels (from 2-bit to 8-bit)
  - Supports different quantization schemes (Q4_K, Q5_K, etc.)
  
- **Implementation**: Converted using llama.cpp tools
- **Best Use Case**: Deployment on consumer hardware or mobile devices

### AWQ
AWQ (Activation-aware Weight Quantization) focuses on preserving the most important weights based on activation patterns.

- **Key Features**:
  - Identifies and preserves weights that have the most impact on activations
  - Applies different quantization strategies based on weight importance
  - Offers exceptional performance at 4-bit precision
  - Faster inference than GPTQ in many cases
  
- **Implementation**: Uses the `awq` library
- **Best Use Case**: When balancing accuracy and speed is critical

### BitsAndBytes
BitsAndBytes is HuggingFace's integration for efficient low-bit transformers implementation.

- **Key Features**:
  - Easy integration with the transformers library
  - Supports Int8 (8-bit) and Int4 (4-bit) quantization
  - Can use NF4 (normalized float 4-bit) format
  - Provides Double Quantization option for further memory savings
  
- **Implementation**: Uses the `bitsandbytes` library with Transformers
- **Best Use Case**: When working in the HuggingFace ecosystem

## Quantization Precision Formats

### Float16 (FP16)
Half-precision floating point format using 16 bits.

- **Structure**: 1 sign bit, 5 exponent bits, 10 mantissa bits
- **Range**: Approximately 6.0 × 10^−8 to 6.5 × 10^4
- **Memory Savings**: 50% compared to FP32
- **Performance Impact**: Minimal degradation for most tasks
- **Hardware Support**: Widely supported on modern GPUs

### Int8
8-bit integer quantization.

- **Structure**: 8-bit signed or unsigned integers
- **Range**: -128 to 127 (signed) or 0 to 255 (unsigned)
- **Memory Savings**: 75% compared to FP32
- **Performance Impact**: Moderate, but acceptable for many applications
- **Hardware Support**: Excellent, including on CPUs and mobile devices

### Int4
4-bit integer quantization.

- **Structure**: 4-bit signed or unsigned integers
- **Range**: -8 to 7 (signed) or 0 to 15 (unsigned)
- **Memory Savings**: 87.5% compared to FP32
- **Performance Impact**: Significant, requires careful implementation
- **Hardware Support**: Limited, but growing with specialized hardware

## Quantization Schemes

### Symmetric Quantization
Maps floating-point values symmetrically around zero.

- **Formula**: `q = round(x / scale)`
- **Dequantization**: `x = q * scale`
- **Advantages**: Simpler, no zero-point required
- **Limitations**: Less efficient for asymmetric distributions

### Asymmetric Quantization
Uses both a scale and zero-point to map floating-point values.

- **Formula**: `q = round(x / scale) + zero_point`
- **Dequantization**: `x = (q - zero_point) * scale`
- **Advantages**: Better utilization of quantization range
- **Limitations**: Slightly more complex, requires storing zero-point

## Model-Specific Quantization Considerations

### Qwen Models

Qwen models (particularly Qwen2.5 series) have specific characteristics that affect quantization performance:

- **Architecture Considerations**:
  - Uses a modified transformer architecture with attention mechanism optimizations
  - Multi-Query Attention (MQA) design requires special handling during quantization
  - Contains Grouped-Query Attention layers which benefit from per-group quantization

- **Quantization Performance**:
  - GPTQ works well with 4-bit precision, showing only minor degradation in code generation tasks
  - AWQ can maintain higher accuracy but requires more calibration data
  - BitsAndBytes quantization is well-supported through HuggingFace's integration

- **Benchmarking Insights**:
  - 14B model can be effectively reduced to ~4GB with 4-bit quantization
  - Code generation tasks (HumanEval) show higher sensitivity to quantization than general text tasks
  - Inference speed benefits substantially from 4-bit quantization (up to 2.5x speedup)

- **Recommended Settings**:
  - For GPTQ: 4-bit with 128 group size, act_order=True for optimal results
  - For AWQ: 4-bit with activation-aware scaling and 128 group size
  - For BitsAndBytes: nf4 format with double quantization for best balance

### Scaling Observations

Recent experiments across model scales (7B to 14B) have revealed consistent patterns:

- Larger models (14B+) tend to be more resilient to 4-bit quantization than smaller models
- Performance retention increases with model size (14B quantized models retain more capabilities than 7B quantized models)
- Memory usage scales approximately linearly with model size at the same quantization level

## Comparative Metrics

### Memory Usage

| Model | Original | FP16 | Int8 | Int4 (GPTQ) | Int4 (AWQ) |
|-------|----------|------|------|-------------|------------|
| Qwen 1.5B | 4.2 GB | 2.3 GB | 1.2 GB | 0.7 GB | 0.8 GB |
| Qwen 7B | 19.3 GB | 9.8 GB | 5.1 GB | 2.6 GB | 2.7 GB |
| Qwen 14B | 38.6 GB | 19.3 GB | 9.6 GB | 4.8 GB | 5.4 GB |