"""Allganize PDF を全部 download + page-level text 抽出して chunks.jsonl 出力."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import pdfplumber
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PDF_DIR = REPO_ROOT / "pdfs"
DOC_CSV = DATA_DIR / "documents.csv"
CHUNKS_OUT = REPO_ROOT / "chunks.jsonl"

PDF_DIR.mkdir(parents=True, exist_ok=True)


def download_pdf(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    try:
        r = requests.get(url, timeout=120, allow_redirects=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"    [download error] {url}: {e}")
        return False


def safe_filename(domain: str, file_name: str) -> str:
    return f"{domain}__{file_name}"


def main() -> None:
    print("=" * 70)
    print("Allganize PDF download + page-level text 抽出")
    print("=" * 70)

    rows = []
    with open(DOC_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Total PDFs: {len(rows)}")

    # 1. download
    print("\n[1/2] PDF download...")
    success_paths: list[tuple[dict, Path]] = []
    for i, row in enumerate(rows):
        domain = row["domain"]
        file_name = row["file_name"]
        url = row["url"]
        local_name = safe_filename(domain, file_name)
        dest = PDF_DIR / local_name
        ok = download_pdf(url, dest)
        if ok:
            size_mb = dest.stat().st_size / 1e6
            print(f"  [{i+1}/{len(rows)}] {local_name} ({size_mb:.2f} MB)")
            success_paths.append((row, dest))
        else:
            print(f"  [{i+1}/{len(rows)}] {local_name} FAILED")

    print(f"\n  Downloaded: {len(success_paths)}/{len(rows)}")

    # 2. parse PDF → chunks (1 chunk = 1 page)
    print("\n[2/2] PDF parse → page-level chunks...")
    n_chunks = 0
    with open(CHUNKS_OUT, "w", encoding="utf-8") as out:
        for i, (row, path) in enumerate(success_paths):
            domain = row["domain"]
            file_name = row["file_name"]
            title = row["title"]
            try:
                with pdfplumber.open(path) as pdf:
                    for page_idx, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        text = text.strip()
                        if len(text) < 30:  # スキップ: ほぼ空 or 図のみ
                            continue
                        chunk = {
                            "domain": domain,
                            "file_name": file_name,
                            "title": title,
                            "page_no": page_idx + 1,  # 1-indexed
                            "doc_id": f"{domain}__{file_name}__p{page_idx+1}",
                            "text": text,
                        }
                        out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        n_chunks += 1
            except Exception as e:
                print(f"  [{i+1}/{len(success_paths)}] {file_name}: parse error {e}")
                continue
            if (i + 1) % 10 == 0:
                print(f"  parsed {i+1}/{len(success_paths)} PDFs, chunks so far: {n_chunks:,}")

    print(f"\n  Total chunks: {n_chunks:,}")
    print(f"  → {CHUNKS_OUT}")


if __name__ == "__main__":
    main()
