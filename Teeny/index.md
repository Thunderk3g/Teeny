---
layout: default
---

# Teeny: Making LLMs Tiny Without Sacrificing Intelligence (v2)

Teeny is a cutting-edge research project exploring the frontier of large language model quantization. We systematically evaluate different quantization techniques to make state-of-the-art AI models run efficiently on resource-constrained hardware while preserving their intelligence and capabilities.

## Quantization Performance Analysis

Our comprehensive research explores the impact of quantization on LLM performance, with particular focus on the trade-offs between model size, inference speed, and task accuracy. Our latest findings show that 4-bit quantization offers the optimal balance for most applications.

![Performance vs Compression](/assets/images/performance_vs_compression.png)

*The graph above illustrates the trade-off between model performance retention and memory reduction across different quantization techniques. Note how AWQ outperforms other methods at equivalent bit-width levels.*

## Key Research Findings

- 4-bit quantization offers the optimal balance between performance retention and size reduction
- AWQ consistently outperforms other methods at equivalent bit-width levels  
- Code generation tasks show higher sensitivity to quantization than general text generation
- Different quantization techniques show varying performance across hardware platforms

## Quantization Methods Compared

We examine three primary quantization approaches, each with distinct advantages:

![Quantization Methods Comparison](/assets/images/radar_comparison.png)

*Radar chart comparing the strengths and weaknesses of different quantization methods across key metrics.*

- **GPTQ (Generalized Post-Training Quantization)**: One-shot weight quantization using approximate second-order information, optimized for GPU inference
- **GGUF (GPT-Generated Unified Format)**: A flexible format designed for CPU inference with optional GPU offloading
- **AWQ (Activation-aware Weight Quantization)**: A technique that protects salient weights by observing activation patterns

Read our detailed [methodology document](./docs/methodology.html) for a comprehensive explanation of our research approach.

## Performance Across Bit Precision

The impact of reducing bit precision varies significantly across different metrics:

![Precision Impact](/assets/images/precision_heatmap.png)

*Heatmap showing how different bit precisions affect model performance, memory usage, and inference speed.*

## Benchmark Results

Our current focus is on quantizing the qwen2.5-instruct-14b model and evaluating its performance on the HumanEval benchmark for code generation.

![Performance Metrics](/assets/images/performance_metrics.png)

*Comparison of memory usage and inference speed across quantization methods.*

| Model Version | HumanEval-Mul (Pass@1) | Memory Usage | Inference Time | Performance Retention |
|---------------|------------------------|--------------|----------------|------------------------|
| Original (FP16) | 45.7%               | 32.5 GB      | 85 tokens/s    | 100.0%                 |
| GPTQ-Int8     | 45.1%                 | 18.2 GB      | 102 tokens/s   | 98.7%                  |
| GPTQ-Int4     | 43.2%                 | 11.3 GB      | 118 tokens/s   | 94.5%                  |
| GGUF-Q4_K_M   | 42.8%                 | 10.8 GB      | 110 tokens/s   | 93.7%                  |
| AWQ-Int4      | 44.1%                 | 11.0 GB      | 125 tokens/s   | 96.5%                  |

*Note: The results above represent our latest benchmark findings.*

See our [benchmark process](./docs/benchmarks.html) for details on our evaluation methodology.

## Implementation Guide

We provide comprehensive instructions for implementing quantization techniques in your own projects. Our [implementation guide](./docs/implementation.html) covers:

- Step-by-step procedures for applying GPTQ, GGUF, and AWQ quantization
- Code examples for quantization, loading, and inference
- Best practices for optimal deployment across different hardware configurations
- Troubleshooting common issues in quantized model performance

## Research Documentation

- [Research Paper](./docs/research.html): Comprehensive analysis of our findings
- [Quantization Methodology](./docs/methodology.html): Detailed explanation of our research approach
- [Benchmark Process](./docs/benchmarks.html): How we evaluate quantized models
- [Implementation Guide](./docs/implementation.html): How to apply these techniques in your own projects

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Thunderk3g/teeny.git
cd teeny

# Install requirements
pip install -r requirements.txt

# Run the benchmark tests
jupyter notebook notebooks/benchmark_tests.ipynb
```

## Why "Teeny"?

The name "Teeny" represents our mission to create extremely small but powerful language models - transforming enormous neural networks into teeny implementations that can run efficiently on resource-constrained hardware without compromising intelligence.

[View on GitHub](https://github.com/Thunderk3g/Teeny)
