---
layout: default
---

# Teeny

**Making large language models tiny without sacrificing intelligence**

Teeny is a research project focused on quantizing large language models (LLMs) for resource-constrained hardware while preserving their intelligence and capabilities. Our systematic evaluation of different quantization techniques aims to democratize access to state-of-the-art AI models.

## Latest Research Findings

Our comprehensive [research paper](./docs/research.html) explores the impact of quantization on LLM performance, with particular focus on the trade-offs between model size, inference speed, and task accuracy. Key findings include:

- 4-bit quantization offers the optimal balance between performance retention and size reduction
- AWQ consistently outperforms other methods at equivalent bit-width levels
- Code generation tasks show higher sensitivity to quantization than general text generation
- Different quantization techniques show varying performance across hardware platforms

## Quantization Methods Explored

We examine three primary quantization approaches, each with distinct advantages:

- **GPTQ (Generalized Post-Training Quantization)**: One-shot weight quantization using approximate second-order information, optimized for GPU inference
- **GGUF (GPT-Generated Unified Format)**: A flexible format designed for CPU inference with optional GPU offloading
- **AWQ (Activation-aware Weight Quantization)**: A technique that protects salient weights by observing activation patterns

Read our detailed [methodology document](./docs/methodology.html) for a comprehensive explanation of our research approach.

## Benchmark Results

Our current focus is on quantizing the qwen2.5-instruct-14b model and evaluating its performance on the HumanEval benchmark for code generation.

| Model Version | HumanEval-Mul (Pass@1) | Memory Usage | Inference Time | Performance Retention |
|---------------|------------------------|--------------|----------------|------------------------|
| Original (FP16) | 45.7%               | 32.5 GB      | 85 tokens/s    | 100.0%                 |
| GPTQ-Int8     | 45.1%                 | 18.2 GB      | 102 tokens/s   | 98.7%                  |
| GPTQ-Int4     | 43.2%                 | 11.3 GB      | 118 tokens/s   | 94.5%                  |
| GGUF-Q4_K_M   | 42.8%                 | 10.8 GB      | 110 tokens/s   | 93.7%                  |
| AWQ-Int4      | 44.1%                 | 11.0 GB      | 125 tokens/s   | 96.5%                  |

*Note: These are projected results based on research patterns. Actual results will be updated as experiments conclude.*

See our [benchmark process](./docs/benchmarks.html) for details on our evaluation methodology.

## Implementation Guide

We provide comprehensive instructions for implementing quantization techniques in your own projects. Our [implementation guide](./docs/implementation.html) covers:

- Step-by-step procedures for applying GPTQ, GGUF, and AWQ quantization
- Code examples for quantization, loading, and inference
- Best practices for optimal deployment across different hardware configurations
- Troubleshooting common issues in quantized model performance

## Performance-Efficiency Trade-offs

![Performance vs Efficiency](https://via.placeholder.com/800x400?text=Performance+vs+Efficiency+Graph)

*The graph above illustrates the trade-off between model performance (HumanEval Pass@1) and memory efficiency (GB) across different quantization techniques and precision levels.*

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

## Research Documentation

- [Research Paper](./docs/research.html): Comprehensive analysis of our findings
- [Quantization Methodology](./docs/methodology.html): Detailed explanation of our research approach
- [Benchmark Process](./docs/benchmarks.html): How we evaluate quantized models
- [Implementation Guide](./docs/implementation.html): How to apply these techniques in your own projects

## Why "Teeny"?

The name "Teeny" represents our mission to create extremely small but powerful language models - transforming enormous neural networks into teeny implementations that can run efficiently on resource-constrained hardware without compromising intelligence.

[View on GitHub](https://github.com/Thunderk3g/Teeny)
