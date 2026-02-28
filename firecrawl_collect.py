import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import requests

FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v1/scrape"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sha1(s: str):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def safe_filename_from_url(url: str, max_len: int = 120):
    base = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in url.strip())
    base = base[:max_len] if len(base) > max_len else base
    return f"{base}__{sha1(url)[:10]}"


def read_url_list(path: str):
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                urls.append(s)
    return urls


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def extract_field(resp: Dict, field: str):
    data = resp.get("data")
    if isinstance(data, dict) and isinstance(data.get(field), str):
        return data[field]
    return resp.get(field, "") if isinstance(resp.get(field), str) else ""


def request_scrape(api_key: str, url: str):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"url": url, "formats": ["markdown"], "onlyMainContent": True, "timeout": 30000}
    r = requests.post(FIRECRAWL_SCRAPE_URL, headers=headers, json=body, timeout=45)
    if r.status_code != 200:
        return False, {"status_code": r.status_code, "text": r.text}
    obj = r.json()
    if obj.get("success") is False:
        return False, obj
    return True, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True)
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    api_key = os.environ.get("FIRECRAWL_API_KEY", "").strip()

    ensure_dir(args.out)
    urls = read_url_list(args.urls)
    if not urls:
        raise SystemExit("No URLs in urls file.")

    ok_count = fail_count = 0
    for url in urls:
        base = safe_filename_from_url(url)
        md_path = os.path.join(args.out, base + ".md")
        json_path = os.path.join(args.out, base + ".json")
        meta_path = os.path.join(args.out, base + ".meta.json")

        ok, resp = request_scrape(api_key, url)
        write_json(json_path, resp)
        write_json(meta_path, {"url": url, "timestamp_utc": now_iso(), "ok": bool(ok)})

        if ok:
            md = extract_field(resp, "markdown")
            if md:
                write_text(md_path, md)
            ok_count += 1
        else:
            fail_count += 1

    print(f"  saved to: {args.out}")
    print(f"  ok      = {ok_count}")
    print(f"  fail    = {fail_count}")


if __name__ == "__main__":
    main()