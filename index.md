---
layout: default
---

# Teeny

**Making large language models tiny without sacrificing intelligence**

Teeny is a research project focused on quantizing large language models (LLMs) for resource-constrained hardware while preserving their intelligence and capabilities.

## Key Features

- **Multiple Quantization Methods**: Comparing GPTQ, GGUF, and AWQ techniques
- **Precision Analysis**: Evaluating Float16, Int8, and other numerical formats
- **Benchmark Results**: Tracking performance on HumanEval and other metrics
- **Practical Implementations**: Ready-to-use code for model quantization

[View on GitHub](https://github.com/your-username/teeny)

## Latest Benchmark Results

Our current focus is on quantizing the qwen2.5-instruct-14b model.

| Model Version | HumanEval-Mul (Pass@1) | Memory Usage | Inference Time |
|---------------|------------------------|--------------|----------------|
| Original      | TBD                    | TBD          | TBD            |
| GPTQ-Int8     | TBD                    | TBD          | TBD            |
| GGUF-Float16  | TBD                    | TBD          | TBD            |
| AWQ           | TBD                    | TBD          | TBD            |

## Documentation

- [Quantization Methodology](./docs/methodology.html)
- [Benchmark Process](./docs/benchmarks.html)
- [Implementation Guide](./docs/implementation.html)
