# Teeny

**Making large language models tiny without sacrificing intelligence**

## Research Objective

This project focuses on the quantization of large pre-trained language models (LLMs) into smaller, highly efficient versions that can run on hardware with minimal resources while maintaining high accuracy. We apply various quantization techniques to pre-trained LLMs and evaluate their performance across multiple dimensions.

## Quantization Methods Explored

- **GPTQ vs GGUF vs AWQ**: Comparative analysis of these popular quantization methods
- **Precision Types**: Float16, Int8, and other numerical formats
- **Quantization Schemes**: Symmetric (zero-centered) and Affine (asymmetric)

## Core Components

### Quantization Fundamentals
- Range mapping
- Affine and Scale Quantization
- Quantization granularity
- Calibration techniques

### Implementation Approaches
- Post Training Quantization (PTQ)
- Weight Quantization
- Activation Quantization
- Partial Quantization
- Quantization Aware Training (QAT)

## Current Focus

Quantizing qwen2.5-instruct-14b to create a model that achieves competitive results on the HumanEval benchmark while requiring significantly fewer resources.

## Benchmark Results

| Model Version | HumanEval-Mul (Pass@1) | Memory Usage | Inference Time |
|---------------|------------------------|--------------|----------------|
| Original      | TBD                    | TBD          | TBD            |
| GPTQ-Int8     | TBD                    | TBD          | TBD            |
| GGUF-Float16  | TBD                    | TBD          | TBD            |
| AWQ           | TBD                    | TBD          | TBD            |

## Repository Structure

- `/notebooks`: Jupyter notebooks with experiments and benchmark tests
- `/models`: Quantized model files or scripts to generate them
- `/results`: Benchmark results and performance comparisons
- `/docs`: Additional documentation on methodologies

## Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/teeny.git
cd teeny

# Install requirements
pip install -r requirements.txt

# Run the benchmark tests
jupyter notebook notebooks/benchmark_tests.ipynb
```

## Why "Teeny"?

The name "Teeny" represents our mission to create extremely small but powerful language models - transforming enormous neural networks into teeny implementations that can run efficiently on resource-constrained hardware without compromising on intelligence.

## References

- [Towards Data Science: GPTQ vs GGUF vs AWQ Comparison](https://medium.com/towards-data-science/which-quantization-method-is-right-for-you-gptq-vs-gguf-vs-awq-c4cd9d77d5be)
- [HuggingFace: Quantization Guide](https://huggingface.co/docs/optimum/en/concept_guides/quantization)
- [Basics of Quantization in ML](https://iq.opengenus.org/basics-of-quantization-in-ml/)
- [Neural Network Compression Using Quantization](https://medium.com/sharechat-techbyte/neural-network-compression-using-quantization-328d22e8855d)
- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
- [HumanEval Benchmark](https://www.datacamp.com/tutorial/humaneval-benchmark-for-evaluating-llm-code-generation-capabilities)
- [LiveCodeBench](https://livecodebench.github.io/)
