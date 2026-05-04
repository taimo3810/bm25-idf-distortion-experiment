"""JaGovFAQs (sbintuitions/JMTEB jagovfaqs_22k) を fetch して chunks.jsonl 出力.

各 FAQ を 1 chunk として扱う。doc_id は連番、tenant は "jagov_general" 固定。
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
JAGOV_OUT = REPO_ROOT / "jagov_chunks.jsonl"


def main() -> None:
    print("=" * 70)
    print("JaGovFAQs fetch (sbintuitions/JMTEB, jagovfaqs_22k)")
    print("=" * 70)

    # corpus のみ取得
    print("\n[1/2] corpus 読み込み...")
    ds = load_dataset("sbintuitions/JMTEB", name="jagovfaqs_22k-corpus", split="corpus")
    print(f"  total: {len(ds):,} docs, fields={list(ds.features.keys())}")
    print(f"  sample: {dict(ds[0])}")

    print("\n[2/2] chunks.jsonl 書き出し...")
    n = 0
    with open(JAGOV_OUT, "w", encoding="utf-8") as f:
        for r in ds:
            doc_id = str(r["docid"])
            text = (r.get("text") or "").strip()
            if not text:
                continue
            rec = {
                "doc_id": f"jagov__{doc_id}",
                "tenant": "jagov_general",
                "text": text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"  written: {n:,} chunks")
    print(f"  → {JAGOV_OUT}")


if __name__ == "__main__":
    main()
