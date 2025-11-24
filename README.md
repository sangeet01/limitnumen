# Numen vs LIMIT Benchmark

This directory contains the benchmark code and results demonstrating Numen's performance on the [**LIMIT dataset**](https://huggingface.co/datasets/orionweller/LIMIT) (DeepMind, 2025). 


##  The Challenge
The LIMIT paper [On the Theoretical Limitations of Embedding-Based Retrieval](https://arxiv.org/abs/2508.21038) argues that dense embedding models have a fundamental theoretical limit based on their dimension ($d$). They showed that even state-of-the-art 7B parameter models (E5-Mistral, GritLM) fail on simple retrieval tasks because they cannot represent all combinations of documents.

##  Numen's Solution
Numen overcomes this limitation by using a **training-free, arbitrary-dimension** approach. Instead of learning a fixed-size embedding (e.g., 4096d) from a finite vocabulary, Numen uses **Character N-Gram Hashing** to map text into a dense vector space of *any* size.

### Key Advantages
1. **Arbitrary Dimension**: Can scale to 8192d, 16384d, or higher without retraining.
2. **No Vocabulary Limit**: Handles any word variations via n-grams.
3. **Zero Training**: Purely algorithmic, no expensive pre-training required.

##  Results (Recall@100)

Numen significantly outperforms SOTA embedding models on the LIMIT benchmark and **beats the sparse BM25 baseline**.

| Model | Type | Dimension | Recall@100 |
|-------|------|-----------|------------|
| **Numen** | **Dense** | **32768** | **93.90%** 🏆 |
| **BM25** | Sparse | ~50k | 93.6% |
| **Numen** | **Dense** | **16384** | **93.05%** |
| **Numen** | **Dense** | **8192** | **89.85%** |
| Promptriever | Dense | 4096 | 18.9% |
| GritLM 7B | Dense | 4096 | 12.9% |
| Gemini Embed | Dense | 3072 | 10.0% |
| E5-Mistral 7B | Dense | 4096 | 8.3% |
| Qwen3 Embed | Dense | 4096 | 4.8% |

**Key Finding**: Numen is the **first dense retrieval model to outperform BM25** (93.9% vs 93.6%) on the LIMIT benchmark, proving that dense vectors can match sparse keyword search without massive vocabularies.

##  How to Run

The benchmark is contained in a single, self-contained Python script designed for Google Colab or local execution.

### Prerequisites
```bash
pip install datasets numpy matplotlib
```

### Running the Benchmark
1. Open `numen.py`.
2. Run the script (it handles dataset downloading via HuggingFace).
3. Results will be printed to stdout and saved to `log.txt`.

### Implementation Details
The script uses a specialized **N-Gram Hashing** mode for Numen:
- **Tokenization**: Character 3-grams, 4-grams, and 5-grams (e.g., "apple" → `^ap`, `app`, `ppl`, `ple`, `le$`).
- **Hashing**: Deterministic CRC32 hash of each n-gram.
- **Vectorization**: Maps hashes to indices in a fixed-size float vector (e.g., 32768d).
- **Similarity**: Cosine similarity between query and document vectors.

##  Files
- [**`numen.ipynb`**](https://github.com/sangeet01/limitnumen/blob/main/limit/numen.ipynb): Complete benchmark notebook (Numen core + LIMIT eval).
- [**`numen.py`**](https://github.com/sangeet01/limitnumen/blob/main/limit/numen.py): Standalone Python script.
- [**`log.txt`**](https://github.com/sangeet01/limitnumen/blob/main/limit/log.txt): Execution log with detailed results.
- [**`numen_.png`**](https://github.com/sangeet01/limitnumen/blob/main/limit/numen_.png): Visualization of performance vs dimension.

##  License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.


##

**PS**: Sangeet's the name, a daft undergrad splashing through chemistry and code like a toddler—my titrations are a mess, and I've used my mouth to pipette. 
