import os
import json
import glob
import re
import unicodedata

RAW_DIR = "data/raw"
OUT_PATH = "data/processed/docs.jsonl"

os.makedirs("data/processed", exist_ok=True)

def strip_bad_unicode(s: str) -> str: # Remove Unicode control/format/surrogate/private-use/unassigned chars
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

def garbage_ratio(s: str) -> float: # ratio of clearly-bad characters
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

def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_ascii_control(text)
    text = strip_bad_unicode(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def load_url_from_meta(md_file: str) -> str:
    meta_file = md_file.replace(".md", ".meta.json")
    if os.path.exists(meta_file):
        try:
            meta = json.load(open(meta_file, "r", encoding="utf-8"))
            return meta.get("url", "") or ""
        except Exception:
            return ""
    return ""

def main():
    kept = 0
    skipped_short = 0
    skipped_garbled = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for md_file in glob.glob(f"{RAW_DIR}/*.md"):
            base = os.path.basename(md_file)
            doc_id = base.replace(".md", "")

            try:
                with open(md_file, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except Exception:
                continue

            text = normalize(text)

            # drop garbage docs
            gr = garbage_ratio(text)
            if gr > 0.002:
                skipped_garbled += 1
                continue

            if len(text) < 300:
                skipped_short += 1
                continue

            url = load_url_from_meta(md_file)

            obj = {
                "doc_id": doc_id,
                "source_url": url,
                "text": text,
            }
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print("Wrote:", OUT_PATH)
    print("Kept docs:", kept)
    print("Skipped short:", skipped_short)
    print("Skipped garbled:", skipped_garbled)

if __name__ == "__main__":
    main()