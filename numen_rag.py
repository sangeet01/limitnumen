

# Cell 0: Setup & Installation (Notebook Only)
# ------------------------------------------------------------------
# This file is designed for Jupyter/Kaggle/Colab notebooks.
# Copy each cell into a notebook cell and run sequentially.
# ------------------------------------------------------------------

!pip install datasets groq numpy faiss-cpu -q
print("Dependencies installed.")

# Download MIRAGE Benchmark
import os
if not os.path.exists("benchmark.json"):
    print("Downloading MIRAGE Benchmark...")
    !wget -q https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json -O benchmark.json
    print("Dataset 'benchmark.json' downloaded.")
else:
    print("Dataset already present.")

print("Setup complete!")


# Cell 1: Imports & Configuration
# ------------------------------------------------------------------
# Numen RAG: A Self-Auditing Medical QA System (Enterprise Scale)
# ------------------------------------------------------------------
import numpy as np
import zlib
import time
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datasets import load_dataset
import warnings

# Try importing FAISS
try:
    import faiss
except ImportError:
    print("FAISS not found. Installing...")
    # Fallback if cell 0 wasn't run
    os.system('pip install faiss-cpu') 
    import faiss

warnings.filterwarnings('ignore')

@dataclass
class Config:
    DIM: int = 32768             # Numen Dimension (High dim for precision)
    NGRAMS: Tuple = (3, 4, 5, 6) # N-gram sizes for lexical matching
    TOP_K: int = 5               # Number of context documents to retrieve
    
    # Verification Thresholds
    # Lowered to 0.4 based on empirical runs (dense vector overlap is lower than exact match)
    HASH_THRESHOLD: float = 0.40     

config = Config()
print("Configuration Loaded (FAISS Scalable Mode).")

# ------------------------------------------------------------------
# GROQ CLIENT SETUP (Rate-Limited)
# ------------------------------------------------------------------
GLOBAL_GROQ_CLIENT = None

class RateLimitedGroqClient:
    """Wrapper for Groq client to enforce rate limits and handle 429 errors."""
    def __init__(self, api_key):
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self.chat = self._Chat(self._client.chat)

    class _Chat:
        def __init__(self, chat_attr):
            self.completions = self._Completions(chat_attr.completions)

        class _Completions:
            def __init__(self, completions_attr):
                self._completions = completions_attr
                self.last_call_time = 0
                self.min_interval = 2.0  # limit ~30 req/min

            def create(self, *args, **kwargs):
                import time
                elapsed = time.time() - self.last_call_time
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
                try:
                    response = self._completions.create(*args, **kwargs)
                    self.last_call_time = time.time()
                    return response
                except Exception as e:
                    if "429" in str(e):
                        print("\n[Shield] 429 Too Many Requests. Cooling down 5 minutes...")
                        time.sleep(300)
                        return self._completions.create(*args, **kwargs)
                    raise e

def initialize_groq_client():
    global GLOBAL_GROQ_CLIENT
    if GLOBAL_GROQ_CLIENT is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("\n" + "="*40 + "\nGROQ API KEY REQUIRED\n" + "="*40)
            try:
                user_key = input("Enter Groq API Key: ").strip()
                if user_key:
                    os.environ["GROQ_API_KEY"] = user_key
                    api_key = user_key
            except: pass
        
        if api_key:
            try:
                GLOBAL_GROQ_CLIENT = RateLimitedGroqClient(api_key=api_key)
                print("[System] Groq Client (Llama-3.3-70B) Initialized.")
            except Exception as e:
                print(f"[System] Groq Init Failed: {e}")
        else:
            print("[System] No Groq Key Found. (Setup Required)")
    return GLOBAL_GROQ_CLIENT


# Cell 2: Layer 1 - Numen Core (Lexical Hashing)
# ------------------------------------------------------------------
class NumenCore:
    """
    Handles fast, training-free vectorization using CRC32 n-gram hashing.
    Used for Retrieval and Entity-Level Hallucination Detection.
    """
    def __init__(self, dim=config.DIM, ngrams=config.NGRAMS):
        self.dim = dim
        self.ngrams = ngrams
        
    def encode(self, text: str) -> np.ndarray:
        text = text.lower().strip()
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text: return vec
        
        count = 0
        for n in self.ngrams:
            if len(text) < n: continue
            for i in range(len(text) - n + 1):
                gram = text[i:i+n]
                # Deterministic Hash
                h = zlib.crc32(gram.encode('utf-8')) & 0xffffffff
                idx = h % self.dim
                # Weighting: Longer n-grams (rare entities) get more weight
                weight = 1.0 + (n - 3) * 0.5 
                vec[idx] += weight
                count += 1
                
        if count > 0:
            vec = np.log1p(vec) # Log-saturation
            norm = np.linalg.norm(vec)
            if norm > 0: vec /= norm
        return vec

print("NumenCore Class Ready.")


# Cell 3: [Removed Semantic Core]
# Per 'finally.txt', we strictly use ONE LLM and Pure Numen Retrieval.
# No BERT/Transformer embeddings needed.


# Cell 4: Numen RAG System v2 (Dense + Chunking + FAISS)
# ------------------------------------------------------------------
class NumenRAG:
    def __init__(self):
        self.numen = NumenCore()
        
        self.index = None   # FAISS Index
        self.documents = [] # Stores chunks
        self.client = None
        
        # Initialize LLM Client
        self.client = initialize_groq_client()

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Splits long text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Try to break at a space
            if end < len(text):
                while end > start and text[end] not in [' ', '\n', '.']:
                    end -= 1
                if end == start: # Force break if no separator
                    end = min(start + chunk_size, len(text))
            
            chunks.append(text[start:end].strip())
            start = end - overlap if end < len(text) else len(text)
            
        return [c for c in chunks if len(c) > 50] # Filter tiny chunks

    def index_data(self, doc_source: Dict[str, str] or List[str]):
        """
        Builds Numen FAISS Index. Accepts Dict {id: text} or List [text].
        Auto-chunks long documents.
        """
        raw_docs = []
        if isinstance(doc_source, dict):
            raw_docs = list(doc_source.values())
        else:
            raw_docs = doc_source
            
        print(f"[Index] Pre-processing {len(raw_docs)} documents...")
        t0 = time.time()
        
        # Chunking Phase
        self.documents = []
        for doc in raw_docs:
            chunks = self.chunk_text(doc)
            self.documents.extend(chunks)
            
        print(f"[Index] Created {len(self.documents)} retrievable chunks.")

        # Batch Encode
        print("[Index] Encoding vectors...")
        # Note: For huge datasets, we would encode in batches.
        # For <1M docs, this is fine in memory.
        matrix = np.zeros((len(self.documents), self.numen.dim), dtype=np.float32)
        for i, text in enumerate(self.documents):
            matrix[i] = self.numen.encode(text)

        # Build FAISS Index
        print("[Index] Building FAISS Index...")
        self.index = faiss.IndexFlatIP(self.numen.dim) # Inner Product (Cosine since normalized)
        self.index.add(matrix)
            
        print(f"[Index] FAISS Index Built in {time.time() - t0:.3f}s. Stored {self.index.ntotal} vectors.")

    def expand_query(self, query: str) -> str:
        """Uses LLM to generate medical keywords for better lexical retrieval."""
        if not self.client: return query
        
        try:
            prompt = f"Given the medical question: '{query}', list 5 key medical terms or synonyms to search for in a textbook. Output ONLY the terms separated by spaces."
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50
            )
            keywords = resp.choices[0].message.content.strip()
            return f"{query} {keywords}"
        except:
            return query

    def retrieve(self, query: str, k=config.TOP_K) -> List[str]:
        """Numen Retrieval (FAISS) with Query Expansion."""
        if not self.index:
            return []
            
        # 1. Expand Query for Semantic-like Lexical overlap
        expanded_q = self.expand_query(query)
        # print(f"    (Expanded Query: {expanded_q[:50]}...)")
        
        q_vec = self.numen.encode(expanded_q).reshape(1, -1) # Resize for FAISS
        scores, indices = self.index.search(q_vec, k)
        
        # Flatten results
        top_indices = indices[0]
        # Return docs, handle -1 (not found)
        return [self.documents[i] for i in top_indices if i >= 0]

    def generate(self, query: str, context: List[str], strict=False) -> str:
        """Generates answer with Citation Enforcement and CoT Reasoning."""
        if not self.client:
            return "[Error] API Client not initialized."

        # v2 Prompt Format
        ctx_str = ""
        for i, doc in enumerate(context):
            ctx_str += f"Source {i+1}:\n{doc}\n\n"
        
        system_msg = """You are an expert medical reasoning engine.
Format your response exactly as follows:
Reasoning: [Think step-by-step]
Final Answer: [The precise answer derived from sources]

1. Analyze the symptoms/question against sources.
2. Cite sources in reasoning (e.g., [Source 1]).
3. If uncertain, say "I don't know"."""
        
        if strict:
            system_msg += " PREVIOUS WAS REJECTED. BE CRITICAL."

        user_msg = f"Sources:\n{ctx_str}\nQuestion: {query}\nAnswer:"

        try:
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.4 # Slightly creative for reasoning
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[Gen Error: {e}]"

print("NumenRAG System v3 (Titan: Expansion + CoT) Ready.")


# Cell 5: The Loop (Lexical + Oracle Verification + Self-Consistency)
# ------------------------------------------------------------------
def run_complete_loop(rag_system, query: str):
    print(f"\n[Question] {query}")
    
    # 1. Retrieve (with Expansion)
    docs = rag_system.retrieve(query)
    
    # 2. Generate (Best of 2 Candidates for Robustness)
    best_answer = ""
    best_score = -1.0
    best_verdict = "UNSAFE"
    
    # Generate 2 candidates to find the most grounded one
    for attempt in range(2): 
        # print(f"  > Generating Candidate {attempt+1}...")
        cand_ans = rag_system.generate(query, docs, strict=(attempt>0))
        
        # Verify Candidate
        ans_vec = rag_system.numen.encode(cand_ans)
        ctx_vec = rag_system.numen.encode(" ".join(docs))
        hash_score = float(np.dot(ans_vec, ctx_vec))
        
        # Oracle Check (Fast check)
        oracle_v = "SAFE"
        if rag_system.client:
            # Simplified Oracle prompt to save tokens
            p = f"Context: {docs[0][:400]}...\nAns: {cand_ans}\nSupported? YES/NO"
            try:
                r = rag_system.client.chat.completions.create(
                    model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": p}]
                )
                if "NO" in r.choices[0].message.content.upper(): oracle_v = "UNSAFE"
            except: pass
            
        # Scoring Logic: Prefer SAFE + High Hash
        # Boost score if SAFE
        final_score = hash_score + (0.5 if oracle_v == "SAFE" else 0.0)
        
        if final_score > best_score:
            best_score = final_score
            best_answer = cand_ans
            best_verdict = oracle_v
            
        # Optimization: If very high score on first try, break early
        if hash_score > 0.6 and oracle_v == "SAFE":
            break

    # Final Decision Output
    print(f"  > [Audit] Best Hash: {best_score:.2f} | Oracle: {best_verdict}")
    print(f"  > [Answer] {best_answer[:100]}...")
        
    return {
        'status': best_verdict,
        'final_answer': best_answer,
        'metrics': best_score
    }

print("Verification Loop Ready.")


# Cell 6: Data Loading & Main Execution (with Accuracy)
# ------------------------------------------------------------------
def load_mirage_data():
    """
    Loads official MIRAGE benchmark data from 'benchmark.json'.
    Returns the full dictionary of datasets.
    """
    import json
    import os
    import urllib.request

    filename = "benchmark.json"
    url = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"

    # 1. auto-download
    if not os.path.exists(filename):
        print(f"[Data] Downloading MIRAGE Benchmark from {url}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(" > Download complete.")
        except Exception as e:
            print(f"[Data] Download failed: {e}")
            return None

    # 2. Load JSON
    try:
        print(f"[Data] Loading MIRAGE Benchmark suite from {filename}...")
        with open(filename, 'r', encoding='utf-8') as f:
            full_benchmark = json.load(f)
            
        print(f" > Benchmark Datasets Found: {list(full_benchmark.keys())}")
        return full_benchmark
        
    except Exception as e:
        print(f"[Data] Error processing benchmark.json: {e}")
        return None

def load_real_pubmed_data():
    """
    Robustly loads PubMedQA from Hugging Face.
    Returns: documents (dict), queries (list of dicts with 'question', 'id', 'truth')
    """
    print("\n[Data] Loading PubMedQA dataset (Standard PQA-L)...")
    try:
        # User request: avoid BigBio/trust_remote_code issues. 
        # Using standard 'pqa_labeled' subset which is clean and safe.
        dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
        print(" > Loaded Standard PubMedQA")
    except Exception as e:
        print(f" > PubMedQA Load Failed: {e}")
        return {}, []
        
    documents = {}
    queries = []
    
    # Limit for demo speed
    MAX_DOCS = 1000 
    TEST_SET_SIZE = 10  # Number of questions to evaluate
    
    print(f"[Data] Processing first {MAX_DOCS} samples...")
    for i, item in enumerate(dataset):
        if i >= MAX_DOCS: break
        
        # Handle Schema
        pid = str(item.get('pubid') or item.get('document_id') or str(i))
        
        # Context
        context_raw = item['context']
        if isinstance(context_raw, dict) and 'contexts' in context_raw:
            text = " ".join(context_raw['contexts'])
        elif isinstance(context_raw, list):
            text = " ".join(context_raw)
        else:
            text = str(context_raw)
            
        documents[pid] = text
        
        # Extract Ground Truth (Long Answer)
        # Standard has 'long_answer', BigBio might use 'answer'
        truth = item.get('long_answer') or item.get('final_decision') or "N/A"
        
        if i < TEST_SET_SIZE: 
            queries.append({
                'id': pid, 
                'question': item['question'],
                'truth': truth
            })
            
    return documents, queries

def evaluate_accuracy(client, system_answer, ground_truth):
    """Uses LLM to judge if system answer matches ground truth."""
    if not client: 
        print("  > [Eval] Skipped (No Client)")
        return False
    
    prompt = f"""Compare these two medical answers.
Ground Truth: {ground_truth}
System Answer: {system_answer}

Are they factually consistent? Reply EXACTLY 'YES' or 'NO'."""
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return "YES" in resp.choices[0].message.content.upper()
    except:
        return False

if __name__ == "__main__":
    print("\n=== Numen RAG: Production Loop (Full MIRAGE Suite) ===")
    
    # 1. Load Full Benchmark
    mirage_suite = load_mirage_data()
    
    if not mirage_suite:
        # Fallback to PubMedQA (Standard) if MIRAGE fails
        print("[System] MIRAGE failed, falling back to Standard PubMedQA...")
        docs, queries = load_real_pubmed_data()
        datasets_to_run = {"Standard_PubMedQA": queries}
    else:
        datasets_to_run = {}
        # Parse MIRAGE into our usable format
        for ds_name, ds_data in mirage_suite.items():
            print(f"\n[Parser] Processing {ds_name}...")
            parsed_queries = []
            
            # Sort keys to ensure deterministic order
            for qid in sorted(ds_data.keys()):
                if len(parsed_queries) >= 5: break # Demo Limit per dataset
                
                item = ds_data[qid]
                q_text = item['question']
                
                # Check for options (MCE) vs Yes/No
                options = item.get('options', {})
                answer_key = item['answer']
                
                if options:
                    # Multiple Choice
                    truth_val = options.get(answer_key, "N/A")
                    truth_str = f"{answer_key}: {truth_val}"
                    # Context for Indexing: Question + Options
                    doc_blob = f"Question: {q_text}\nOptions:\n" + \
                               "\n".join([f"{k}: {v}" for k,v in options.items()])
                else:
                    # Yes/No (PubMedQA/BioASQ in MIRAGE format)
                    truth_str = answer_key # likely "yes", "no", "maybe"
                    # Context for Indexing: Question only (Self-retrieval)
                    doc_blob = f"Question: {q_text}"

                parsed_queries.append({
                    'id': qid,
                    'question': q_text,
                    'truth': truth_str,
                    'doc_context': doc_blob 
                })
            
            datasets_to_run[ds_name] = parsed_queries

    # 2. Execution Loop across all datasets
    rag = NumenRAG()
    
    for ds_name, test_queries in datasets_to_run.items():
        print(f"\n" + "="*60)
        print(f"BENCHMARK: {ds_name.upper()}")
        print(f"="*60)
        
        if not test_queries:
            print(f"[Warn] No queries found for {ds_name}.")
            continue
            
        # Build Index specific for this dataset
        # In a real run, we might have one massive index, but for the demo 
        # we index the 'documents' (Question+Options) of the current test set
        current_docs = [q['doc_context'] for q in test_queries]
        rag.index_data(current_docs)
        
        correct = 0
        for q_item in test_queries:
            q_text = q_item['question']
            truth = q_item['truth']
            
            # Run Pipeline
            result = run_complete_loop(rag, q_text)
            
            # Eval
            is_correct = evaluate_accuracy(rag.client, result['final_answer'], truth)
            if is_correct:
                correct += 1
                print(f"  > [Eval] PASS")
            else:
                print(f"  > [Eval] FAIL (Expected: {truth})")
            print("-" * 40)
            
        print(f"\n[Score] {ds_name}: {correct}/{len(test_queries)} ({(correct/len(test_queries))*100:.1f}%)")

    print(f"\n[System] Full Benchmark Compliance Run Complete.")
