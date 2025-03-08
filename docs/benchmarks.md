---
layout: default
title: Benchmark Process
---

# Benchmark Process

## Evaluation Framework

The Teeny project employs a comprehensive benchmarking framework to assess the impact of quantization on large language models (LLMs). Our evaluation protocol is designed to measure both the performance degradation and efficiency gains resulting from various quantization techniques.

## Core Benchmarks

### HumanEval

The primary benchmark used in our evaluation is **HumanEval**, a collection of Python programming problems designed to test the model's ability to generate functionally correct code. HumanEval is particularly relevant for our research as it:

- Tests the model's reasoning capabilities
- Requires maintaining precision in logical operations
- Provides unambiguous correctness criteria
- Allows for pass@k evaluation metrics

Our implementation uses the HumanEval-Mul variant, which extends the original benchmark with multi-line reasoning tasks.

#### Metrics

For HumanEval, we report:

- **Pass@1**: The percentage of problems solved correctly on the first attempt
- **Pass@10**: The percentage of problems where at least one correct solution appears in 10 samples

### Memory Usage

We measure the memory footprint of each quantized model using:

- **Peak VRAM**: Maximum GPU memory consumption during inference
- **Peak RAM**: Maximum system memory usage for CPU-based inference
- **Model Size**: Storage footprint of the quantized model weights

All measurements are reported in gigabytes (GB) and compared against the baseline unquantized model.

### Inference Speed

Inference speed is measured across different hardware configurations:

1. **Token Generation Rate**: Tokens per second during text generation
2. **Latency**: Time to first token and inter-token latency
3. **Throughput**: Number of inference requests processed per unit time

Tests are conducted with:
- Batch size: 1 (single inference) and 32 (batch inference)
- Input context: 512 tokens
- Generation length: 128 tokens

## Hardware Specifications

All benchmarks are run on standardized hardware configurations:

- **High-end GPU**: NVIDIA A100 (80GB)
- **Consumer GPU**: NVIDIA RTX 4090 (24GB)
- **Edge Device**: NVIDIA Jetson AGX Orin (64GB)
- **CPU**: Intel Xeon Platinum 8380 (2.3 GHz, 40 cores)
- **Apple Silicon**: M2 Ultra (64GB unified memory)

## Experimental Protocol

Our benchmarking methodology follows these steps:

1. **Baseline Establishment**:
   - Run all benchmarks on the original unquantized model
   - Record performance metrics as reference points

2. **Controlled Environment**:
   - Identical software stacks across all tests
   - Temperature monitoring to prevent thermal throttling
   - Isolated execution environment to prevent resource contention

3. **Statistical Robustness**:
   - Each experiment is repeated 5 times
   - Results are reported with mean and standard deviation
   - Outlier detection and removal is applied

4. **Performance-Efficiency Tradeoff Analysis**:
   - Plotting accuracy degradation vs. memory reduction
   - Computing efficiency metrics (accuracy per watt, accuracy per GB)

## Comparative Analysis

For each quantization technique (GPTQ, GGUF, AWQ), we evaluate:

1. **Performance Degradation**: Percentage drop in accuracy relative to the original model
2. **Memory Reduction**: Percentage reduction in memory usage
3. **Speed Improvement**: Percentage change in inference speed
4. **Energy Efficiency**: Power consumption during inference

## Visualization Methods

Our benchmark results are presented using:

1. **Performance-vs-Compression Plots**: Showing the relationship between model accuracy and size reduction
2. **Radar Charts**: Comparing multiple metrics across different quantization techniques
3. **Hardware-specific Performance Profiles**: Showing how each technique performs across different hardware

## Current Results

Our current benchmarking efforts focus on the qwen2.5-instruct-14b model with the following quantization techniques:

| Model Version | HumanEval-Mul (Pass@1) | Memory Usage | Inference Time |
|---------------|------------------------|--------------|----------------|
| Original      | TBD                    | TBD          | TBD            |
| GPTQ-Int8     | TBD                    | TBD          | TBD            |
| GGUF-Float16  | TBD                    | TBD          | TBD            |
| AWQ           | TBD                    | TBD          | TBD            |

*Updated results will be published as our research progresses.*

## Reproducibility

All benchmark code, environment specifications, and raw results are available in the [repository](https://github.com/Thunderk3g/Teeny/tree/main/benchmarks) to ensure reproducibility of our findings.

## Limitations

We acknowledge the following limitations in our current benchmarking approach:

1. Focus on a single domain (code generation) may not generalize to all LLM use cases
2. Hardware-specific optimizations may affect the relative performance of different techniques
3. Variations in implementation details of quantization algorithms may impact results

Future work will address these limitations by expanding our benchmark suite and hardware configurations. 