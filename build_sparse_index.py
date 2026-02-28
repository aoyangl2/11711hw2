import json
import os
import pickle
import re
from rank_bm25 import BM25Okapi

CHUNKS_PATH = "data/processed/chunks.jsonl"
OUT_BM25 = "data/processed/bm25.pkl"
OUT_IDS = "data/processed/bm25_ids.json"

STOP = {
    "the","a","an","is","was","were","to","of","and","in","on","for","at","by","from",
    "what","when","who","where","how","which","that","this","these","those","as","it",
    "are","be","been","being","or","with","into","about","their","its"
}

def tok(text: str):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if t not in STOP]

def main():
    os.makedirs("data/processed", exist_ok=True)

    tokenized_docs = []
    ids = []

    MIN_CHARS = 30

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("chunk_id", "")
            txt = (obj.get("text") or "").strip()
            if not cid or not txt:
                continue
            if len(txt) < MIN_CHARS:
                continue

            ids.append(cid)
            tokenized_docs.append(tok(txt))

    bm25 = BM25Okapi(tokenized_docs)

    with open(OUT_BM25, "wb") as f:
        pickle.dump(bm25, f)

    with open(OUT_IDS, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)

    print("Wrote:", OUT_BM25, "and", OUT_IDS)
    print("Docs:", len(ids))

    avg_len = sum(len(d) for d in tokenized_docs) / len(tokenized_docs)
    print(f"Average token count per chunk: {avg_len:.2f}")

if __name__ == "__main__":
    main()