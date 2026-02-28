import json
import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed/chunks.jsonl"
OUT_INDEX = "data/processed/faiss.index"
OUT_IDS = "data/processed/faiss_ids.json"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 64 

MIN_CHARS = 30

def main():
    os.makedirs("data/processed", exist_ok=True)

    ids = []
    texts = []
    total = 0
    skipped_empty = 0
    skipped_short = 0

    print(f"Reading chunks from {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            obj = json.loads(line)
            cid = obj.get("chunk_id", "")
            txt = (obj.get("text", "") or "").strip()

            if not cid or not txt:
                skipped_empty += 1
                continue
            if len(txt) < MIN_CHARS:
                skipped_short += 1
                continue

            ids.append(cid)
            texts.append(txt)

    print(f"Total chunks:  {total}")
    print(f"Kept chunks:   {len(texts)}")
    print(f"Skipped empty: {skipped_empty}")
    print(f"Skipped short: {skipped_short}")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    elif torch.backends.mps.is_available():
        model = model.to("mps")

    print(f"Encoding {len(texts)} chunks (batch_size={BATCH_SIZE})...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    print("Embeddings generated. Shape:", embeddings.shape)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, OUT_INDEX)
    with open(OUT_IDS, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)

    print(f"Successfully wrote index to {OUT_INDEX}")
    print(f"Successfully wrote IDs to {OUT_IDS}")

if __name__ == "__main__":
    main()