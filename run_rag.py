import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

CHUNKS_PATH = "data/processed/chunks.jsonl"
FAISS_INDEX_PATH = "data/processed/faiss.index"
FAISS_IDS_PATH = "data/processed/faiss_ids.json"
BM25_PKL_PATH = "data/processed/bm25.pkl"
BM25_IDS_PATH = "data/processed/bm25_ids.json"

EMBEDDER_NAME = "BAAI/bge-large-en-v1.5"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEN_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

STOP_BM25 = {
    "the","a","an","is","was","were","to","of","and","in","on","for","at","by","from",
    "what","when","who","where","how","which","that","this","these","those","as","it",
    "are","be","been","being","or","with","into","about","their","its"
}

YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_queries(path: str) -> Dict[str, str]:
    obj = json.load(open(path, "r", encoding="utf-8"))

    if isinstance(obj, dict) and "queries" in obj and isinstance(obj["queries"], list):
        obj = obj["queries"]
    if isinstance(obj, dict) and "questions" in obj and isinstance(obj["questions"], list):
        obj = obj["questions"]

    if isinstance(obj, list):
        out: Dict[str, str] = {}
        for i, item in enumerate(obj):
            if isinstance(item, str):
                out[str(i + 1)] = item
            elif isinstance(item, dict):
                qid = item.get("id", str(i + 1))
                q = item.get("question", item.get("query", item.get("text", "")))
                out[str(qid)] = q if isinstance(q, str) else str(q)
            else:
                out[str(i + 1)] = str(item)
        return out

    if isinstance(obj, dict):
        out: Dict[str, str] = {}
        for k, v in obj.items():
            if isinstance(v, str):
                out[str(k)] = v
            elif isinstance(v, dict):
                q = v.get("question", v.get("query", v.get("text", "")))
                out[str(k)] = q if isinstance(q, str) else str(q)
            else:
                out[str(k)] = str(v)
        return out

    raise ValueError(f"Unrecognized query file format: {type(obj)}")


def load_chunks(path: str) -> Dict[str, Dict[str, str]]:
    store: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj["chunk_id"]
            store[cid] = {
                "text": obj.get("text", "") or "",
                "url": obj.get("url", obj.get("source_url", "")) or "",
                "title": obj.get("title", obj.get("doc_title", "")) or "",
            }
    return store


# Text utils
def bm25_tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if t not in STOP_BM25]


def clean_context(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[\d+\]", "", text)
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if "skip to main content" in low or "jump to content" in low:
            continue
        if len(s) < 2:
            continue
        lines.append(s)
    return "\n".join(lines)


def cap_per_source_ids(ids: List[str], store: Dict[str, Dict[str, str]], cap: int = 1) -> List[str]:
    seen = defaultdict(int)
    out: List[str] = []
    for cid in ids:
        url = (store.get(cid, {}) or {}).get("url", "").strip()
        if seen[url] < cap:
            out.append(cid)
            seen[url] += 1
    return out


# Retrieval
def dense_retrieve(
    query: str,
    embedder: SentenceTransformer,
    index: faiss.Index,
    faiss_ids: List[str],
    k: int,
    embedder_name: str,
) -> List[str]:
    name = (embedder_name or "").lower()
    if "bge" in name:
        qtext = "Represent this sentence for searching relevant passages: " + query
    elif "e5" in name:
        qtext = "query: " + query
    else:
        qtext = query

    q = embedder.encode([qtext], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q, k)

    out: List[str] = []
    for i in idxs[0]:
        ii = int(i)
        if 0 <= ii < len(faiss_ids):
            out.append(faiss_ids[ii])
    return out


def bm25_retrieve(query: str, bm25: BM25Okapi, bm25_ids: List[str], k: int) -> List[str]:
    toks = bm25_tokenize(query)
    scores = bm25.get_scores(toks)
    topk = np.argsort(scores)[::-1][:k]
    return [bm25_ids[int(i)] for i in topk]


# Generation
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_generator(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    device = pick_device()
    mdl = mdl.to(device).eval()
    return mdl, tok, device


def make_prompt(question: str, context: str, tokenizer) -> str:
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an expert on Carnegie Mellon University and Pittsburgh. "
                "Answer the user's question directly and confidently. "
                "Use the provided context as a primary reference, but if the information is missing, "
                "use your internal knowledge to provide the most accurate and complete answer possible. "
                "Do not mention that the context is insufficient. Do not say 'I don't know' or 'unknown'."
            )
        },
        {
            "role": "user", 
            "content": f"Context:\n{context}\n\nProvide the answer only, no irrelevant reasoning or explanation.Question: {question}"
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def postprocess_answer(ans: str) -> str:
    a = (ans or "").strip()
    if not a:
        return "unknown"
    a = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", a)      # strip markdown links
    a = re.sub(r"https?://\S+", "", a).strip()          # strip raw urls
    a = re.sub(r"^(Answer)[:：]\s*", "", a, flags=re.I)  # strip "Answer:"
    a = re.sub(r"\s+", " ", a).strip()
    return a if a else "unknown"


# RAG
def rag_answer(
    question: str,
    store: Dict[str, Dict[str, str]],
    embedder: SentenceTransformer,
    faiss_index: faiss.Index,
    faiss_ids: List[str],
    bm25: BM25Okapi,
    bm25_ids: List[str],
    reranker: CrossEncoder,
    gen_model,
    gen_tok,
    gen_device: torch.device,
    k_dense: int,
    k_sparse: int,
    k_ctx_dense: int,
    k_ctx_sparse: int,
    gen_max_input_tokens: int,
    gen_max_new_tokens: int,
) -> str:
    d_ids = dense_retrieve(question, embedder, faiss_index, faiss_ids, k_dense, EMBEDDER_NAME)
    s_ids = bm25_retrieve(question, bm25, bm25_ids, k_sparse)

    candidate_ids = list(dict.fromkeys(d_ids[: min(10, len(d_ids))] + s_ids[: min(10, len(s_ids))]))
    if candidate_ids:
        pairs = [[question, (store.get(cid, {}) or {}).get("text", "")] for cid in candidate_ids]
        scores = reranker.predict(pairs)
        ranked = [cid for _, cid in sorted(zip(scores, candidate_ids), key=lambda x: x[0], reverse=True)]
    else:
        ranked = list(dict.fromkeys(d_ids[:k_ctx_dense] + s_ids[:k_ctx_sparse]))

    ranked = cap_per_source_ids(ranked, store, cap=1)[:6]

    ctxs: List[str] = []
    for cid in ranked:
        txt = clean_context((store.get(cid, {}) or {}).get("text", ""))
        if txt:
            ctxs.append(txt)

    context = "\n\n".join(ctxs)[:1500]
    if not context.strip():
        return "unknown"

    prompt = make_prompt(question, context, gen_tok)
    raw = generate_answer(gen_model, gen_tok, gen_device, prompt, gen_max_input_tokens, gen_max_new_tokens)
    return postprocess_answer(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k_dense", type=int, default=60)
    ap.add_argument("--k_sparse", type=int, default=80)
    ap.add_argument("--k_ctx_dense", type=int, default=3)
    ap.add_argument("--k_ctx_sparse", type=int, default=2)
    ap.add_argument("--gen_max_input_tokens", type=int, default=1024)
    ap.add_argument("--gen_max_new_tokens", type=int, default=100)
    args = ap.parse_args()

    store = load_chunks(CHUNKS_PATH)
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    faiss_ids = json.load(open(FAISS_IDS_PATH, "r", encoding="utf-8"))
    bm25 = pickle.load(open(BM25_PKL_PATH, "rb"))
    bm25_ids = json.load(open(BM25_IDS_PATH, "r", encoding="utf-8"))

    embedder = SentenceTransformer(EMBEDDER_NAME)
    reranker = CrossEncoder(RERANKER_NAME, device=pick_device())

    gen_model, gen_tok, gen_device = build_generator(GEN_MODEL_NAME)

    qmap = load_queries(args.queries)
    preds: Dict[str, str] = {"andrewid": "aoyangl", "timestamp_utc": now_iso()}

    for qid, question in qmap.items():
        ans = rag_answer(
            question=question,
            store=store,
            embedder=embedder,
            faiss_index=faiss_index,
            faiss_ids=faiss_ids,
            bm25=bm25,
            bm25_ids=bm25_ids,
            reranker=reranker,
            gen_model=gen_model,
            gen_tok=gen_tok,
            gen_device=gen_device,
            k_dense=args.k_dense,
            k_sparse=args.k_sparse,
            k_ctx_dense=args.k_ctx_dense,
            k_ctx_sparse=args.k_ctx_sparse,
            gen_max_input_tokens=args.gen_max_input_tokens,
            gen_max_new_tokens=args.gen_max_new_tokens,
        )
        preds[str(qid)] = ans
        print(f"[{qid}] {question}\n -> {ans}\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()