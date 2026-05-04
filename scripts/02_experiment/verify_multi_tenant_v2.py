"""異業種マルチテナント SaaS の IDF 歪み実験 v2.

v1 からの差分:
  - クエリは LLM 生成「サポートデスク問い合わせ風」(generated_queries.json)
  - 各 query の relevant doc は target_chunk_doc_id 1 件 (binary qrel)
  - NDCG@10 を正しい idcg ベースで計算 (relevant 1 件 / k=10 なら ideal=1.0)
  - per-query Recall@50, NDCG@10, MRR の分布 (mean/median/p5/p95/min/max) を出力
  - IDF gap × harm の対応データも results に含める

Usage:
    uv run python experiments/idf-distortion-multi-tenant/verify_multi_tenant_v2.py
    uv run python experiments/idf-distortion-multi-tenant/verify_multi_tenant_v2.py --skip-index --keep-indices
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

load_dotenv()

ES_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
ES_API_KEY = os.environ.get("ELASTICSEARCH_API_KEY")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
ALLGANIZE_CHUNKS = REPO_ROOT / "chunks.jsonl"
JAGOV_CHUNKS = REPO_ROOT / "jagov_chunks.jsonl"
QUERIES_IN = DATA_DIR / "generated_queries.json"
RESULTS_OUT = RESULTS_DIR / "results_multi_tenant_v2.json"

ALLGANIZE_TENANTS = ["finance", "it", "manufacturing", "public", "retail"]
NOISE_TENANT = "jagov_general"

IDX_MIXED = "idf-multi-tenant-mixed"
IDX_AONLY_PREFIX = "idf-multi-tenant-aonly-"

INDEX_BODY: dict[str, Any] = {
    "settings": {
        "analysis": {
            "analyzer": {
                "ja_analyzer": {
                    "type": "custom",
                    "tokenizer": "kuromoji_tokenizer",
                    "filter": ["kuromoji_baseform", "lowercase"],
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "text": {"type": "text", "analyzer": "ja_analyzer"},
            "tenant": {"type": "keyword"},
            "file_name": {"type": "keyword"},
            "page_no": {"type": "integer"},
        }
    },
}


def load_chunks() -> list[dict]:
    chunks: list[dict] = []
    with open(ALLGANIZE_CHUNKS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            chunks.append({
                "doc_id": r["doc_id"], "tenant": r["domain"],
                "file_name": r["file_name"], "page_no": r.get("page_no"),
                "text": r["text"],
            })
    with open(JAGOV_CHUNKS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            chunks.append({
                "doc_id": r["doc_id"], "tenant": NOISE_TENANT,
                "file_name": r["doc_id"], "page_no": None, "text": r["text"],
            })
    return chunks


def setup_indices(es: Elasticsearch, chunks: list[dict]) -> None:
    all_indices = [IDX_MIXED] + [IDX_AONLY_PREFIX + t for t in ALLGANIZE_TENANTS]
    for idx in all_indices:
        es.indices.delete(index=idx, ignore_unavailable=True)
        es.indices.create(index=idx, body=INDEX_BODY)

    def gen(idx: str, filter_tenant: str | None = None):
        for c in chunks:
            if filter_tenant and c["tenant"] != filter_tenant:
                continue
            yield {"_index": idx, "_id": c["doc_id"], "_source": c}

    print(f"  bulk → {IDX_MIXED} ({len(chunks):,} chunks)")
    helpers.bulk(es, gen(IDX_MIXED), chunk_size=1000, request_timeout=600)
    for t in ALLGANIZE_TENANTS:
        idx = IDX_AONLY_PREFIX + t
        n = sum(1 for c in chunks if c["tenant"] == t)
        print(f"  bulk → {idx} ({n:,} chunks)")
        helpers.bulk(es, gen(idx, t), chunk_size=1000, request_timeout=600)
    for idx in all_indices:
        es.indices.refresh(index=idx)
        n = es.count(index=idx)["count"]
        print(f"    {idx}: {n:,} docs")


def run_search(es: Elasticsearch, index: str, query_text: str,
               filter_tenant: str | None = None, size: int = 50) -> list[dict]:
    if filter_tenant:
        body = {
            "query": {"bool": {
                "must": [{"match": {"text": query_text}}],
                "filter": [{"term": {"tenant": filter_tenant}}],
            }},
            "size": size,
        }
    else:
        body = {"query": {"match": {"text": query_text}}, "size": size}
    r = es.search(index=index, body=body, search_type="dfs_query_then_fetch")
    return [{"id": h["_id"], "score": h["_score"]} for h in r["hits"]["hits"]]


# === metrics (binary qrel: target chunk 1 件 only) ===
def find_rank(returned: list[dict], target_id: str) -> int:
    """1-based rank。見つからなければ 0."""
    for i, r in enumerate(returned):
        if r["id"] == target_id:
            return i + 1
    return 0


def ndcg_at_k(returned: list[dict], target_id: str, k: int) -> float:
    rank = find_rank(returned[:k], target_id)
    if rank == 0:
        return 0.0
    # binary qrel, 1 件のみ relevant. ideal は rank=1
    return (1.0 / math.log2(rank + 1)) / 1.0


def recall_at_k(returned: list[dict], target_id: str, k: int) -> float:
    rank = find_rank(returned[:k], target_id)
    return 1.0 if rank > 0 else 0.0


def mrr(returned: list[dict], target_id: str) -> float:
    rank = find_rank(returned, target_id)
    return 1.0 / rank if rank > 0 else 0.0


def rbo(list1: list[str], list2: list[str], p: float = 0.9) -> float:
    s1, s2 = set(), set()
    sum_term = 0.0
    n = min(len(list1), len(list2))
    for d in range(n):
        s1.add(list1[d])
        s2.add(list2[d])
        sum_term += (len(s1 & s2) / (d + 1)) * (p ** d)
    return (1 - p) * sum_term


def summarize(values: list[float], precision: int = 4) -> dict:
    if not values:
        return {"n": 0}
    sv = sorted(values)
    n = len(sv)
    return {
        "n": n,
        "mean": round(statistics.fmean(values), precision),
        "median": round(sv[n // 2], precision),
        "p5": round(sv[max(0, int(n * 0.05) - 1)], precision),
        "p25": round(sv[max(0, int(n * 0.25) - 1)], precision),
        "p75": round(sv[min(n - 1, int(n * 0.75))], precision),
        "p95": round(sv[min(n - 1, int(n * 0.95))], precision),
        "min": round(min(values), precision),
        "max": round(max(values), precision),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-indices", action="store_true")
    ap.add_argument("--skip-index", action="store_true")
    args = ap.parse_args()

    es = Elasticsearch(ES_URL, api_key=ES_API_KEY) if ES_API_KEY else Elasticsearch(ES_URL)
    info = es.info()
    print(f"ES {info['version']['number']}")

    print("\n[1/4] chunks 統合 + queries 読み込み...")
    chunks = load_chunks()
    n_per_tenant = Counter(c["tenant"] for c in chunks)
    for t, n in sorted(n_per_tenant.items()):
        print(f"    {t}: {n:,}")
    with open(QUERIES_IN, encoding="utf-8") as f:
        qdata = json.load(f)
    queries = qdata["queries"]
    print(f"  生成 query 数: {len(queries)}")

    if not args.skip_index:
        print("\n[2/4] index 構築...")
        setup_indices(es, chunks)
    else:
        print("\n[2/4] [skip] index 構築スキップ")

    print("\n[3/4] 実験実行...")
    results: list[dict] = []
    for i, q in enumerate(queries):
        if i % 20 == 0 and i > 0:
            print(f"    {i:,}/{len(queries):,}...")
        d = q["tenant"]
        target_id = q["target_chunk_doc_id"]
        ids_mixed = run_search(es, IDX_MIXED, q["query"], filter_tenant=d, size=50)
        ids_aonly = run_search(es, IDX_AONLY_PREFIX + d, q["query"], filter_tenant=None, size=50)

        m_ndcg10 = ndcg_at_k(ids_mixed, target_id, 10)
        a_ndcg10 = ndcg_at_k(ids_aonly, target_id, 10)
        m_recall50 = recall_at_k(ids_mixed, target_id, 50)
        a_recall50 = recall_at_k(ids_aonly, target_id, 50)
        m_mrr = mrr(ids_mixed, target_id)
        a_mrr = mrr(ids_aonly, target_id)
        rbo50 = rbo([r["id"] for r in ids_mixed[:50]], [r["id"] for r in ids_aonly[:50]])

        results.append({
            "tenant": d,
            "term": q["term"],
            "query": q["query"],
            "target_chunk_doc_id": target_id,
            "gap": q["gap"],
            "subset_count": q["subset_count"],
            "global_count": q["global_count"],
            "mixed_rank": find_rank(ids_mixed, target_id),
            "aonly_rank": find_rank(ids_aonly, target_id),
            "mixed_ndcg@10": round(m_ndcg10, 4),
            "aonly_ndcg@10": round(a_ndcg10, 4),
            "mixed_recall@50": round(m_recall50, 4),
            "aonly_recall@50": round(a_recall50, 4),
            "mixed_mrr": round(m_mrr, 4),
            "aonly_mrr": round(a_mrr, 4),
            "rbo@50": round(rbo50, 4),
            "harm_recall50": round(a_recall50 - m_recall50, 4),
            "harm_ndcg10": round(a_ndcg10 - m_ndcg10, 4),
            "harm_mrr": round(a_mrr - m_mrr, 4),
        })

    # === 集計 ===
    print("\n[4/4] 集計...\n")
    print(f"対象 query 数: {len(results)}")
    keys = ["mixed_recall@50", "aonly_recall@50", "harm_recall50",
            "mixed_ndcg@10", "aonly_ndcg@10", "harm_ndcg10",
            "mixed_mrr", "aonly_mrr", "harm_mrr", "rbo@50"]
    print(f"\n{'metric':<20s} {'mean':>9s} {'med':>9s} {'p5':>9s} {'p25':>9s} {'p75':>9s} {'p95':>9s} {'min':>9s} {'max':>9s}")
    for key in keys:
        d = summarize([r[key] for r in results])
        print(f"  {key:<18s} {d['mean']:+9.4f} {d['median']:+9.4f} {d['p5']:+9.4f} {d['p25']:+9.4f} {d['p75']:+9.4f} {d['p95']:+9.4f} {d['min']:+9.4f} {d['max']:+9.4f}")

    print("\n  tenant ごと harm_recall50 mean:")
    by_tenant: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_tenant[r["tenant"]].append(r["harm_recall50"])
    for t, vals in sorted(by_tenant.items()):
        print(f"    {t}: mean={statistics.fmean(vals):+.4f}, n={len(vals)}, max={max(vals):+.4f}")

    # harm 集計分析
    n_harm_pos = sum(1 for r in results if r["harm_recall50"] > 0)
    n_harm_zero = sum(1 for r in results if r["harm_recall50"] == 0)
    n_harm_neg = sum(1 for r in results if r["harm_recall50"] < 0)
    print(f"\n  harm_recall50 distribution:")
    print(f"    > 0 (mixed が劣る = 過小評価 harm): {n_harm_pos} ({n_harm_pos/len(results)*100:.1f}%)")
    print(f"    = 0: {n_harm_zero} ({n_harm_zero/len(results)*100:.1f}%)")
    print(f"    < 0 (mixed が優れる): {n_harm_neg} ({n_harm_neg/len(results)*100:.1f}%)")

    # harm 大きい query top-15
    sorted_by_harm = sorted(results, key=lambda r: r["harm_recall50"], reverse=True)
    print("\n  harm 大きい query top-15 (Recall@50 ベース):")
    for r in sorted_by_harm[:15]:
        print(f"    [{r['tenant']}] '{r['query']}' (term: {r['term']}, gap: {r['gap']:.1f})")
        print(f"      mixed_rank={r['mixed_rank']}, aonly_rank={r['aonly_rank']}, "
              f"recall_mixed={r['mixed_recall@50']}, recall_aonly={r['aonly_recall@50']}, harm={r['harm_recall50']:+.3f}")

    # gap × harm 散布図相当データ
    print("\n  IDF gap ごとの平均 harm:")
    by_gap_bin: dict[str, list[float]] = defaultdict(list)
    for r in results:
        gap = r["gap"]
        if gap < 5: bin_label = "<5"
        elif gap < 7: bin_label = "5-7"
        elif gap < 10: bin_label = "7-10"
        elif gap < 15: bin_label = "10-15"
        else: bin_label = "≥15"
        by_gap_bin[bin_label].append(r["harm_recall50"])
    for bl in ["<5", "5-7", "7-10", "10-15", "≥15"]:
        if bl in by_gap_bin:
            vals = by_gap_bin[bl]
            print(f"    gap {bl}: mean_harm={statistics.fmean(vals):+.4f}, n={len(vals)}, max={max(vals):+.4f}")

    out = {
        "es_version": info["version"]["number"],
        "n_chunks": len(chunks),
        "chunks_per_tenant": dict(n_per_tenant),
        "n_queries": len(queries),
        "summary": {key: summarize([r[key] for r in results]) for key in keys},
        "harm_buckets": {
            "positive_count": n_harm_pos,
            "zero_count": n_harm_zero,
            "negative_count": n_harm_neg,
        },
        "results": results,
    }
    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n→ 保存: {RESULTS_OUT}")

    if not args.keep_indices:
        for t in ALLGANIZE_TENANTS:
            es.indices.delete(index=IDX_AONLY_PREFIX + t, ignore_unavailable=True)
        es.indices.delete(index=IDX_MIXED, ignore_unavailable=True)
        print("indices deleted (use --keep-indices to retain)")


if __name__ == "__main__":
    main()
