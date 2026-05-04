"""Direction B (過大評価) 実験 — Multi-tenant 構成.

Direction A 版 (verify_multi_tenant_v2.py) と同じデータセット構成 + 評価ロジック。
クエリは generated_queries_dir_b.json から読み込む。
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
QUERIES_IN = DATA_DIR / "generated_queries_dir_b.json"
RESULTS_OUT = RESULTS_DIR / "results_multi_tenant_dir_b.json"

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


def find_rank(returned: list[dict], target_id: str) -> int:
    for i, r in enumerate(returned):
        if r["id"] == target_id:
            return i + 1
    return 0


def ndcg_at_k(rank: int, k: int) -> float:
    if rank == 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def recall_at_k(rank: int, k: int) -> float:
    return 1.0 if 0 < rank <= k else 0.0


def mrr(rank: int) -> float:
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


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    sv = sorted(values)
    n = len(sv)
    return {
        "n": n,
        "mean": round(statistics.fmean(values), 4),
        "median": round(sv[n // 2], 4),
        "p5": round(sv[max(0, int(n * 0.05) - 1)], 4),
        "p25": round(sv[max(0, int(n * 0.25) - 1)], 4),
        "p75": round(sv[min(n - 1, int(n * 0.75))], 4),
        "p95": round(sv[min(n - 1, int(n * 0.95))], 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
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

    print("\n[3/4] 実験実行 (K=5 ベース)...")
    K = 5
    results: list[dict] = []
    for i, q in enumerate(queries):
        if i % 10 == 0 and i > 0:
            print(f"    {i:,}/{len(queries):,}...")
        d = q["tenant"]
        target_id = q["target_chunk_doc_id"]
        ids_mixed = run_search(es, IDX_MIXED, q["query"], filter_tenant=d, size=50)
        ids_aonly = run_search(es, IDX_AONLY_PREFIX + d, q["query"], filter_tenant=None, size=50)

        m_rank = find_rank(ids_mixed, target_id)
        a_rank = find_rank(ids_aonly, target_id)

        m_ndcg5 = ndcg_at_k(m_rank, K)
        a_ndcg5 = ndcg_at_k(a_rank, K)
        m_recall5 = recall_at_k(m_rank, K)
        a_recall5 = recall_at_k(a_rank, K)
        m_mrr = mrr(m_rank)
        a_mrr = mrr(a_rank)
        rbo50 = rbo([r["id"] for r in ids_mixed[:50]], [r["id"] for r in ids_aonly[:50]])

        results.append({
            "tenant": d,
            "term": q["term"],
            "query": q["query"],
            "target_chunk_doc_id": target_id,
            "gap": q["gap"],
            "subset_count": q["subset_count"],
            "global_count": q["global_count"],
            "n_other_chunks_with_term": q["n_other_chunks_with_term"],
            "mixed_rank": m_rank,
            "aonly_rank": a_rank,
            "mixed_ndcg@5": round(m_ndcg5, 4),
            "aonly_ndcg@5": round(a_ndcg5, 4),
            "mixed_recall@5": round(m_recall5, 4),
            "aonly_recall@5": round(a_recall5, 4),
            "mixed_mrr": round(m_mrr, 4),
            "aonly_mrr": round(a_mrr, 4),
            "rbo@50": round(rbo50, 4),
            "harm_recall5": round(a_recall5 - m_recall5, 4),
            "harm_ndcg5": round(a_ndcg5 - m_ndcg5, 4),
            "harm_mrr": round(a_mrr - m_mrr, 4),
        })

    print("\n[4/4] 集計...\n")
    print(f"対象 query 数: {len(results)}")
    print(f"\n{'metric':<20s} {'mean':>9s} {'median':>9s} {'p5':>9s} {'p25':>9s} {'p75':>9s} {'p95':>9s} {'min':>9s} {'max':>9s}")
    keys = ["mixed_recall@5", "aonly_recall@5", "harm_recall5",
            "mixed_ndcg@5", "aonly_ndcg@5", "harm_ndcg5",
            "mixed_mrr", "aonly_mrr", "harm_mrr", "rbo@50"]
    for key in keys:
        d = summarize([r[key] for r in results])
        print(f"  {key:<18s} {d['mean']:+9.4f} {d['median']:+9.4f} {d['p5']:+9.4f} {d['p25']:+9.4f} {d['p75']:+9.4f} {d['p95']:+9.4f} {d['min']:+9.4f} {d['max']:+9.4f}")

    print("\n  tenant ごと harm_ndcg5 mean:")
    by_tenant_n: dict[str, list[float]] = defaultdict(list)
    by_tenant_r: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_tenant_n[r["tenant"]].append(r["harm_ndcg5"])
        by_tenant_r[r["tenant"]].append(r["harm_recall5"])
    for t in sorted(by_tenant_n):
        nv = by_tenant_n[t]
        rv = by_tenant_r[t]
        print(f"    {t:>14s}: n={len(nv):3d}, mean_harm_ndcg5={statistics.fmean(nv):+.4f}, "
              f"max_harm_ndcg5={max(nv):+.3f}, mean_harm_recall5={statistics.fmean(rv):+.4f}, "
              f"n_pos_ndcg5={sum(1 for v in nv if v > 0)}, n_pos_recall5={sum(1 for v in rv if v > 0)}")

    n_harm_pos_ndcg = sum(1 for r in results if r["harm_ndcg5"] > 0)
    n_harm_zero_ndcg = sum(1 for r in results if r["harm_ndcg5"] == 0)
    n_harm_neg_ndcg = sum(1 for r in results if r["harm_ndcg5"] < 0)
    n_harm_pos_rec = sum(1 for r in results if r["harm_recall5"] > 0)
    n_harm_zero_rec = sum(1 for r in results if r["harm_recall5"] == 0)
    n_harm_neg_rec = sum(1 for r in results if r["harm_recall5"] < 0)

    print(f"\n  harm 方向の分布 (件数, 全 {len(results)} query):")
    print(f"    harm_ndcg5  : pos={n_harm_pos_ndcg:3d} ({n_harm_pos_ndcg/len(results)*100:5.1f}%), zero={n_harm_zero_ndcg:3d}, neg={n_harm_neg_ndcg:3d}")
    print(f"    harm_recall5: pos={n_harm_pos_rec:3d} ({n_harm_pos_rec/len(results)*100:5.1f}%), zero={n_harm_zero_rec:3d}, neg={n_harm_neg_rec:3d}")

    print("\n  harm 大きい query top-10 (harm_ndcg5 順):")
    for r in sorted(results, key=lambda x: x["harm_ndcg5"], reverse=True)[:10]:
        print(f"    [{r['tenant']}] '{r['query']}' (term={r['term']}, gap={r['gap']:.1f}, n_other={r['n_other_chunks_with_term']})")
        print(f"      rank: mixed={r['mixed_rank']:>3d}, aonly={r['aonly_rank']:>3d} | "
              f"NDCG@5: mixed={r['mixed_ndcg@5']}, aonly={r['aonly_ndcg@5']} (harm={r['harm_ndcg5']:+.3f}) | "
              f"Recall@5: harm={r['harm_recall5']:+.0f}")

    pushed_out = [r for r in results if r["mixed_rank"] > 5 and 0 < r["aonly_rank"] <= 5]
    print(f"\n  mixed が top-5 から押し出された query: {len(pushed_out)} / {len(results)} ({len(pushed_out)/len(results)*100:.1f}%)")
    for r in pushed_out:
        print(f"    [{r['tenant']}] '{r['query']}' (term={r['term']}, gap={r['gap']:.1f}) "
              f"mixed_rank={r['mixed_rank']}, aonly_rank={r['aonly_rank']}")

    out = {
        "es_version": info["version"]["number"],
        "n_chunks": len(chunks),
        "chunks_per_tenant": dict(n_per_tenant),
        "n_queries": len(queries),
        "summary_metrics": {key: summarize([r[key] for r in results]) for key in keys},
        "harm_buckets": {
            "ndcg5": {"positive": n_harm_pos_ndcg, "zero": n_harm_zero_ndcg, "negative": n_harm_neg_ndcg},
            "recall5": {"positive": n_harm_pos_rec, "zero": n_harm_zero_rec, "negative": n_harm_neg_rec},
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
