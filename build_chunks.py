import json
import os
import re
import unicodedata
from typing import List

IN_PATH = "data/processed/docs.jsonl"
OUT_PATH = "data/processed/chunks.jsonl"

CHUNK_SIZE = 450
OVERLAP = 80

os.makedirs("data/processed", exist_ok=True)

def strip_bad_unicode(s: str) -> str:
    if not s:
        return ""
    out = []
    for ch in s:
        if ch in ("\n", "\t"):
            out.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat[0] == "C":
            continue
        out.append(ch)
    return "".join(out)

def strip_ascii_control(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", s)
    s = re.sub(r"\\u00(0[0-8bcef]|1[0-9a-f])", " ", s, flags=re.IGNORECASE)
    return s

def garbage_ratio(s: str) -> float:
    if not s:
        return 1.0
    bad = 0
    for ch in s:
        if ch == "\ufffd":
            bad += 1
            continue
        if ch not in ("\n", "\t"):
            cat = unicodedata.category(ch)
            if cat[0] == "C":
                bad += 1
    return bad / max(1, len(s))

def clean_doc_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"For a list of browsers that this site supports,.*?\.", "", text, flags=re.IGNORECASE)
    text = text.replace("Skip to main content", "")
    text = text.replace("\\\\", ", ")

    noise_patterns = [
        r"Warning: You are viewing this site with an outdated/unsupported browser.*?\.",
        r"Please update your browser.*?\.",
        r"Skip to content",
        r"Check My Account",
        r"Contact Us",
        r"Privacy Policy",
        r"All rights reserved"
    ]
    for p in noise_patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.DOTALL)

    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = text.replace("**", "").replace("*", "")
    text = re.sub(r"\w+\s+>\s+\w+", " ", text) 
    text = re.sub(r'https?://\S+', '', text)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")
def sentence_split(text: str) -> List[str]:
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    sents: List[str] = []
    for p in paras:
        parts = _SENT_SPLIT.split(p)
        for s in parts:
            ss = s.strip()
            # if len(ss) < 40:
            #     continue
            sents.append(ss)
    return sents

def chunk_text(text: str) -> List[str]:
    sents = sentence_split(text)
    chunks: List[str] = []
    cur: List[str] = []
    words = 0

    for s in sents:
        w = len(s.split())
        if cur and words + w > CHUNK_SIZE:
            chunk = " ".join(cur).strip()
            if chunk:
                chunks.append(chunk)

            tail_words = chunk.split()[-OVERLAP:]
            tail = " ".join(tail_words).strip()
            cur = [tail] if tail else []
            words = len(tail_words)

        cur.append(s)
        words += w

    if cur:
        chunk = " ".join(cur).strip()
        if chunk:
            chunks.append(chunk)

    return chunks

def main():
    total_docs = 0
    total_chunks = 0
    skipped_docs = 0
    skipped_chunks_garbled = 0

    with open(IN_PATH, "r", encoding="utf-8") as f, open(OUT_PATH, "w", encoding="utf-8") as out:
        for line in f:
            total_docs += 1
            doc = json.loads(line)

            clean = clean_doc_text(doc.get("text", ""))

            if len(clean) < 300 or garbage_ratio(clean) > 0.002:
                skipped_docs += 1
                continue

            chunks = chunk_text(clean)
            chunks = [c for c in chunks if len(c) >= 200]

            for i, c in enumerate(chunks):
                if garbage_ratio(c) > 0.002:
                    skipped_chunks_garbled += 1
                    continue

                obj = {
                    "chunk_id": f'{doc["doc_id"]}_{i}',
                    "doc_id": doc["doc_id"],
                    "source_url": doc.get("source_url", ""),
                    "text": c,
                }
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total_chunks += 1

    print("Wrote:", OUT_PATH)
    print("Docs read:", total_docs)
    print("Docs skipped:", skipped_docs)
    print("Chunks written:", total_chunks)
    print("Chunks skipped (garbled):", skipped_chunks_garbled)

if __name__ == "__main__":
    main()