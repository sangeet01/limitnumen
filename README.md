# Numen vs LIMIT Benchmark

This directory contains the benchmark code and results demonstrating Numen's performance on the **LIMIT dataset** (DeepMind, 2025).

##  The Challenge
The LIMIT paper [On the Theoretical Limitations of Embedding-Based Retrieval](https://arxiv.org/abs/2508.21038) argues that dense embedding models have a fundamental theoretical limit based on their dimension ($d$). They showed that even state-of-the-art 7B parameter models (E5-Mistral, GritLM) fail on simple retrieval tasks because they cannot represent all combinations of documents.

##  Numen's Solution
Numen overcomes this limitation by using a **training-free, arbitrary-dimension** approach. Instead of learning a fixed-size embedding (e.g., 4096d) from a finite vocabulary, Numen uses **Character N-Gram Hashing** to map text into a dense vector space of *any* size.

### Key Advantages
1. **Arbitrary Dimension**: Can scale to 8192d, 16384d, or higher without retraining.
2. **No Vocabulary Limit**: Handles any word variations via n-grams.
3. **Zero Training**: Purely algorithmic, no expensive pre-training required.

##  Results (Recall@100)

Numen significantly outperforms SOTA embedding models on the LIMIT benchmark.

| Model | Type | Dimension | Recall@100 |
|-------|------|-----------|------------|
| **Numen** | **Dense** | **8192** | **88.40%** 🏆 |
| **Numen** | **Dense** | **4096** | **82.65%** |
| **Numen** | **Dense** | **2048** | **74.75%** |
| Promptriever | Dense | 4096 | 18.9% |
| **Numen** | **Dense** | **1024** | **49.05%** |
| GritLM 7B | Dense | 4096 | 12.9% |
| Gemini Embed | Dense | 3072 | 10.0% |
| E5-Mistral 7B | Dense | 4096 | 8.3% |
| Qwen3 Embed | Dense | 4096 | 4.8% |
| BM25 | Sparse | ~50k | 93.6% |

**Note**: BM25 (sparse) is included as a baseline. Numen is the top-performing **dense** model, bridging the gap between traditional embeddings and sparse retrieval.

##  How to Run

The benchmark is contained in a single, self-contained Python script designed for Google Colab or local execution.

### Prerequisites
```bash
pip install datasets numpy matplotlib
```

### Running the Benchmark
1. Open `one.py`.
2. Run the script (it handles dataset downloading via HuggingFace).
3. Results will be printed to stdout and saved to `log.txt`.

### Implementation Details
The script uses a specialized **N-Gram Hashing** mode for Numen:
- **Tokenization**: Character 3-grams and 4-grams (e.g., "apple" → `^ap`, `app`, `ppl`, `ple`, `le$`).
- **Hashing**: Deterministic CRC32 hash of each n-gram.
- **Vectorization**: Maps hashes to indices in a fixed-size float vector (e.g., 8192d).
- **Similarity**: Cosine similarity between query and document vectors.

##  Files
- `one.ipynb`: Complete benchmark script (Numen core + LIMIT eval).
- `log.txt`: Execution log with detailed results.
- `numen_vs_limit.png`: Visualization of performance vs dimension.



##

**PS**: Sangeet's the name, a daft undergrad splashing through chemistry and code like a toddler—my titrations are a mess, and I've used my mouth to pipette. 






