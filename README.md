# RAG System Execution Guide

This document outlines the step-by-step pipeline for data collection, indexing, and running the Retrieval-Augmented Generation (RAG) system for the CMU and Pittsburgh historical domain.

## Prerequisites

Python 3.9+

Firecrawl API Key (set as an environment variable)

GPU access recommended for embedding and generation tasks

## Execution Pipeline

Follow these steps in order to build the knowledge base and generate predictions.

**Step 1: Data Collection**

Collect raw web pages using the Firecrawl API. This script utilizes a priority queue to scrape URLs and their sub-links.

```
python firecrawl_collect.py \
  --urls urls.txt \
  --out data/raw
```

**Step 2: Document Processing**
Normalize the raw HTML/Markdown files and perform initial denoising.

```
python build_docs.py
```

**Step 3: Text Chunking**
Segment the processed documents into overlapping windows (450 tokens with 80-token overlap) to prepare for indexing.

```
python build_chunks.py
```

**Step 4: Indexing**

Build both Dense (Vector-based) and Sparse (Keyword-based) indices to enable hybrid retrieval.

Dense Index (FAISS + BGE-Large):

```
python build_dense_index.py
```

Sparse Index (BM25):

```
python build_sparse_index.py
```

**Step 5: Run Inference**
Execute the RAG pipeline to answer queries. This script fuses the hybrid retrieval results, applies a Cross-Encoder re-ranker, and generates answers using the Qwen-2.5-14B model.

```
python run_rag.py \
    --queries leaderboard_queries.json \
    --out data/predictions.json \
    --andrewid aoyangl \
    --k_dense 60 \
    --k_sparse 80 \
    --k_ctx_dense 3 \
    --k_ctx_sparse 2 \
    --gen_max_input_tokens 1024 \
    --gen_max_new_tokens 64
```
