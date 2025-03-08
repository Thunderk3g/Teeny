---
layout: default
title: Quantization Methodology
---

# Quantization Methodology

## Overview

This document outlines the scientific methodology employed in the Teeny project for quantizing large language models (LLMs) while preserving their performance. Our research approach is systematic and empirical, focusing on the trade-offs between model size, computational efficiency, and task performance.

## Research Questions

Our methodology is designed to address several key research questions:

1. How do different quantization techniques (GPTQ, GGUF, AWQ) impact model performance across various benchmarks?
2. What is the relationship between numerical precision (e.g., FP16, INT8, INT4) and model capability?
3. Can activation-aware approaches outperform traditional weight quantization methods?
4. Is there an optimal quantization strategy for specific hardware constraints?
5. How does model size affect the performance degradation observed during quantization?

## Experimental Framework

Our experimental methodology follows a rigorous scientific process:

### 1. Model Selection

We carefully select base models that represent state-of-the-art performance in relevant tasks. For our current research, we focus on:

- **qwen2.5-instruct-14b**: A powerful instruction-tuned model that demonstrates excellent performance on code generation tasks

### 2. Quantization Technique Implementation

We systematically apply multiple quantization techniques to each model:

- **GPTQ (Generalized Post-Training Quantization)**: One-shot weight quantization based on approximate second-order information, primarily for GPU acceleration
- **GGUF (GPT-Generated Unified Format)**: A format designed for CPU inference with optional GPU offloading
- **AWQ (Activation-aware Weight Quantization)**: A technique that protects salient weights by observing activations during quantization

Each technique is applied with varying precision levels:

- Float16 (16-bit)
- Int8 (8-bit)
- Int4 (4-bit)
- Int2 (2-bit, for extreme compression)

### 3. Calibration Dataset Selection

Model quantization requires calibration data to determine optimal quantization parameters. Our methodology uses:

- A diverse set of examples representing the target task domain
- Careful balance between domain-specific and general text
- Sufficient sample size to capture the statistical distribution of weights and activations

### 4. Evaluation Protocol

All quantized models undergo rigorous evaluation against the original model:

- **Benchmarks**: HumanEval for code generation, with emphasis on functional correctness
- **Hardware Metrics**: Memory usage, inference time on target hardware
- **Performance Degradation Analysis**: Statistical analysis of performance drop relative to original model

### 5. Error Analysis

We conduct detailed error analysis on failed examples to understand:
- Types of reasoning failures introduced by quantization
- Pattern-based categorization of errors
- Relationship between error types and quantization methods

## Quantization Implementation Details

### Weight Quantization

For weight quantization, we map floating-point weights to discrete integer values using:

1. **Range Analysis**: Determining min/max values for weight clipping
2. **Scaling Factor Computation**: Computing optimal scale factors for each tensor
3. **Quantization Scheme Application**: 
   - Symmetric quantization (zero-centered)
   - Asymmetric quantization (with zero-point offset)

### Granularity Levels

Our research explores multiple quantization granularities:

- **Tensor-wise**: One scale factor per weight tensor
- **Channel-wise**: One scale factor per output channel
- **Group-wise**: One scale factor per group of channels
- **Mixed-precision**: Variable bit-width assignment based on sensitivity analysis

### Activation Quantization

For techniques that quantize activations, we employ:

- Dynamic range adaptation during inference
- Batch normalization folding
- Layer-specific calibration strategies

## Comparative Analysis Framework

Our methodology includes a systematic comparison framework:

1. **Performance vs. Compression Trade-off Analysis**
2. **Hardware-specific Optimization Strategies**
3. **Task-specific Quantization Sensitivity Analysis**
4. **Statistical Significance Testing of Results**

## Reproducibility Measures

To ensure scientific rigor and reproducibility:

- All experiments are conducted with fixed random seeds
- Hardware specifications are precisely documented
- Implementation details are provided in code repositories
- Evaluation scripts and data preparation code are published

## Research Ethics

Our research methodology includes consideration of:

- Energy consumption measurement during training and inference
- Environmental impact of model deployment strategies
- Transparency in reporting limitations and failure cases

This methodology document will be updated as our research progresses and new techniques are incorporated into our experimental framework. 