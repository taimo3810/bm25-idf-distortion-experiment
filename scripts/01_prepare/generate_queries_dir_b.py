"""Direction B 候補単語 × chunk から LLM でサポートデスク問い合わせ風 query を生成.

Direction A と同じプロンプト構造。生成 query は:
  - 過大評価単語 (= 業種特有語) を必ず含む
  - target chunk のユニーク属性語も含める (= ノイズ chunks との distinguishability)
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CANDIDATES_IN = DATA_DIR / "direction_b_candidates.json"
QUERIES_OUT = DATA_DIR / "generated_queries_dir_b.json"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY が無い")

MODEL = "gpt-4o-mini"
N_PER_CAND = 2
CONCURRENCY = 8

TENANT_LABEL = {
    "finance": "金融",
    "it": "IT",
    "manufacturing": "製造業",
    "public": "公共・行政",
    "retail": "小売",
}

PROMPT_TMPL = """あなたは「{tenant_label}」業種のサポートデスク担当者です。社内ナレッジベースから以下のドキュメントを探したい場面を想定してください。

# ドキュメント (該当ページ抜粋)
{chunk_text}

# 条件
- このドキュメントを発見するための、サポートデスク担当者が typing しそうな**自然な短い検索クエリ**を 2 件生成してください。
- 各クエリは **「{term}」というキーワードを必ず含める**こと。
- ドキュメントの内容を識別するための **discriminative な別の単語** (固有名詞、技術用語、サービス名など) も併記すること。
- 各クエリは **15〜40 文字** に収める (簡潔・口語可)。
- 「教えてください」「ですか」のような冗長な疑問形式は避け、検索ボックスに打ち込む実用的な短文 or 短い疑問文にしてください。
- 各クエリはドキュメントの内容と直接対応する検索意図を持つこと。

# 出力形式 (純粋に 2 行、各行が 1 クエリ。番号や記号は付けない)
クエリ1
クエリ2"""


async def gen_one(client: AsyncOpenAI, cand: dict) -> list[str]:
    prompt = PROMPT_TMPL.format(
        tenant_label=TENANT_LABEL[cand["tenant"]],
        chunk_text=cand["chunk_text"],
        term=cand["term"],
    )
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=120,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  [error] {cand['tenant']}/{cand['term']}: {e}")
        return []
    queries = []
    for line in text.strip().split("\n"):
        line = line.strip(" 　\t-・*0123456789.)")
        if not line:
            continue
        if cand["term"] not in line:
            continue
        if not (8 <= len(line) <= 80):
            continue
        queries.append(line)
    return queries[:N_PER_CAND]


async def main_async() -> None:
    with open(CANDIDATES_IN, encoding="utf-8") as f:
        data = json.load(f)
    candidates = data["candidates"]
    print(f"候補数: {len(candidates)}")

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def wrapped(cand):
        async with sem:
            return cand, await gen_one(client, cand)

    tasks = [wrapped(c) for c in candidates]
    out_records = []
    n_done = 0
    for fut in asyncio.as_completed(tasks):
        cand, queries = await fut
        n_done += 1
        if n_done % 5 == 0:
            print(f"  {n_done}/{len(candidates)} done")
        for q in queries:
            out_records.append({
                "tenant": cand["tenant"],
                "term": cand["term"],
                "query": q,
                "target_file_name": cand["chunk_file_name"],
                "target_chunk_doc_id": cand["chunk_doc_id"],
                "subset_count": cand["subset_count"],
                "global_count": cand["global_count"],
                "gap": cand["gap"],
                "n_other_chunks_with_term": cand["n_other_chunks_with_term"],
            })

    print(f"\n生成 query 総数: {len(out_records)}")
    print("\nサンプル (各業種から 3 件):")
    by_tenant: dict[str, list[dict]] = {}
    for r in out_records:
        by_tenant.setdefault(r["tenant"], []).append(r)
    for t, rs in by_tenant.items():
        print(f"  [{t}] {len(rs)} queries")
        for r in rs[:3]:
            print(f"    '{r['query']}' (term: {r['term']}, gap: {r['gap']})")

    with open(QUERIES_OUT, "w", encoding="utf-8") as f:
        json.dump({"n_queries": len(out_records), "queries": out_records}, f, ensure_ascii=False, indent=2)
    print(f"\n→ {QUERIES_OUT}")


if __name__ == "__main__":
    asyncio.run(main_async())
