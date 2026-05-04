"""v2 結果を K=5 中心で再分析 + レポート用データ生成.

K=5 の根拠: 実 RAG ではユーザーの context window 制約から top-5 を LLM に入れるのが現実的。
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_IN = RESULTS_DIR / "results_multi_tenant_v2.json"
ANALYSIS_OUT = RESULTS_DIR / "analysis_k5.json"


def ndcg_at_k(rank: int, k: int) -> float:
    """binary qrel (1 件のみ relevant)、ideal は rank=1。"""
    if rank == 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def recall_at_k(rank: int, k: int) -> float:
    return 1.0 if 0 < rank <= k else 0.0


def agg(values: list[float]) -> dict:
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
    with open(RESULTS_IN, encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]
    K = 5

    # K=5 ベースで全 metrics 再計算
    for r in results:
        m_rank = r["mixed_rank"]
        a_rank = r["aonly_rank"]
        r["mixed_recall@5"] = recall_at_k(m_rank, K)
        r["aonly_recall@5"] = recall_at_k(a_rank, K)
        r["mixed_ndcg@5"] = round(ndcg_at_k(m_rank, K), 4)
        r["aonly_ndcg@5"] = round(ndcg_at_k(a_rank, K), 4)
        r["harm_recall5"] = r["aonly_recall@5"] - r["mixed_recall@5"]
        r["harm_ndcg5"] = round(r["aonly_ndcg@5"] - r["mixed_ndcg@5"], 4)

    print("=" * 70)
    print("K=5 ベース集計 (LLM context window 想定の現実的 K)")
    print("=" * 70)

    print(f"\n総 query 数: {len(results)}")

    # === 集計テーブル ===
    print(f"\n{'metric':<20s} {'mean':>9s} {'median':>9s} {'p5':>9s} {'p95':>9s} {'min':>9s} {'max':>9s}")
    keys_main = ["mixed_recall@5", "aonly_recall@5", "harm_recall5",
                 "mixed_ndcg@5", "aonly_ndcg@5", "harm_ndcg5",
                 "mixed_mrr", "aonly_mrr", "harm_mrr"]
    summary = {}
    for key in keys_main:
        d = agg([r[key] for r in results])
        summary[key] = d
        print(f"  {key:<18s} {d['mean']:+9.4f} {d['median']:+9.4f} {d['p5']:+9.4f} {d['p95']:+9.4f} {d['min']:+9.4f} {d['max']:+9.4f}")

    # === harm 方向の分布 (件数) ===
    print("\n=== harm 方向の分布 (件数, 全 128 query) ===")
    bucket_summary = {}
    for k in ["recall5", "ndcg5", "mrr"]:
        pos = sum(1 for r in results if r[f"harm_{k}"] > 0)
        zero = sum(1 for r in results if r[f"harm_{k}"] == 0)
        neg = sum(1 for r in results if r[f"harm_{k}"] < 0)
        bucket_summary[k] = {"positive": pos, "zero": zero, "negative": neg}
        print(f"  harm_{k:<10s}: positive (mixed 劣) = {pos:3d} ({pos/len(results)*100:5.1f}%), "
              f"zero = {zero:3d}, negative (mixed 優) = {neg:3d} ({neg/len(results)*100:5.1f}%)")

    # === gap × harm_ndcg5 ===
    print("\n=== gap × harm_ndcg5 ===")
    by_gap_bin: dict[str, list[float]] = defaultdict(list)
    by_gap_recall: dict[str, list[float]] = defaultdict(list)
    for r in results:
        gap = r["gap"]
        if gap < 5: bl = "<5"
        elif gap < 7: bl = "5-7"
        elif gap < 10: bl = "7-10"
        elif gap < 15: bl = "10-15"
        else: bl = "≥15"
        by_gap_bin[bl].append(r["harm_ndcg5"])
        by_gap_recall[bl].append(r["harm_recall5"])
    gap_summary = {}
    for bl in ["<5", "5-7", "7-10", "10-15", "≥15"]:
        if bl in by_gap_bin:
            ndcg_vals = by_gap_bin[bl]
            recall_vals = by_gap_recall[bl]
            gap_summary[bl] = {
                "n": len(ndcg_vals),
                "mean_harm_ndcg5": round(statistics.fmean(ndcg_vals), 4),
                "max_harm_ndcg5": round(max(ndcg_vals), 4),
                "mean_harm_recall5": round(statistics.fmean(recall_vals), 4),
                "n_pos_ndcg5": sum(1 for v in ndcg_vals if v > 0),
                "n_pos_recall5": sum(1 for v in recall_vals if v > 0),
            }
            print(f"  gap {bl:>6s}: n={len(ndcg_vals):3d}, "
                  f"mean_harm_ndcg5={statistics.fmean(ndcg_vals):+.4f}, "
                  f"mean_harm_recall5={statistics.fmean(recall_vals):+.4f}, "
                  f"n_pos (ndcg/recall) = {gap_summary[bl]['n_pos_ndcg5']}/{gap_summary[bl]['n_pos_recall5']}")

    # === tenant × harm_ndcg5 ===
    print("\n=== tenant × harm_ndcg5 ===")
    by_tenant_n: dict[str, list[float]] = defaultdict(list)
    by_tenant_r: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_tenant_n[r["tenant"]].append(r["harm_ndcg5"])
        by_tenant_r[r["tenant"]].append(r["harm_recall5"])
    tenant_summary = {}
    for t in sorted(by_tenant_n):
        nv = by_tenant_n[t]
        rv = by_tenant_r[t]
        tenant_summary[t] = {
            "n": len(nv),
            "mean_harm_ndcg5": round(statistics.fmean(nv), 4),
            "max_harm_ndcg5": round(max(nv), 4),
            "mean_harm_recall5": round(statistics.fmean(rv), 4),
            "n_pos_ndcg5": sum(1 for v in nv if v > 0),
            "n_pos_recall5": sum(1 for v in rv if v > 0),
        }
        print(f"  {t:>14s}: n={len(nv):3d}, "
              f"mean_harm_ndcg5={statistics.fmean(nv):+.4f}, max={max(nv):+.3f}, "
              f"n_pos_ndcg5={tenant_summary[t]['n_pos_ndcg5']:2d}, n_pos_recall5={tenant_summary[t]['n_pos_recall5']:2d}")

    # === harm 大きい query top-15 (NDCG@5 ベース) ===
    print("\n=== harm 大きい query top-15 (harm_ndcg5 順) ===")
    top_harm = []
    for r in sorted(results, key=lambda x: x["harm_ndcg5"], reverse=True)[:15]:
        top_harm.append({
            "tenant": r["tenant"],
            "query": r["query"],
            "term": r["term"],
            "gap": r["gap"],
            "mixed_rank": r["mixed_rank"],
            "aonly_rank": r["aonly_rank"],
            "mixed_ndcg5": r["mixed_ndcg@5"],
            "aonly_ndcg5": r["aonly_ndcg@5"],
            "harm_ndcg5": r["harm_ndcg5"],
            "harm_recall5": r["harm_recall5"],
        })
        print(f"  [{r['tenant']}] '{r['query']}' (term={r['term']}, gap={r['gap']:.1f})")
        print(f"    rank: mixed={r['mixed_rank']:>3d}, aonly={r['aonly_rank']:>3d} | "
              f"NDCG@5: mixed={r['mixed_ndcg@5']}, aonly={r['aonly_ndcg@5']} (harm={r['harm_ndcg5']:+.3f}) | "
              f"Recall@5: harm={r['harm_recall5']:+.0f}")

    # === mixed_rank > 5 で aonly_rank ≤ 5 (= top-5 から押し出された) ===
    pushed_out = [r for r in results if r["mixed_rank"] > 5 and 0 < r["aonly_rank"] <= 5]
    print(f"\n=== mixed が top-5 から押し出された query (aonly は top-5 内) ===")
    print(f"  total: {len(pushed_out)} / {len(results)} ({len(pushed_out)/len(results)*100:.1f}%)")
    for r in pushed_out:
        print(f"  [{r['tenant']}] '{r['query']}' (term={r['term']}, gap={r['gap']:.1f}) "
              f"mixed_rank={r['mixed_rank']}, aonly_rank={r['aonly_rank']}")

    # === 出力 ===
    out = {
        "k": K,
        "n_queries": len(results),
        "summary_metrics": summary,
        "harm_buckets": bucket_summary,
        "by_gap": gap_summary,
        "by_tenant": tenant_summary,
        "top_harm_queries": top_harm,
        "pushed_out_top5": [{
            "tenant": r["tenant"], "query": r["query"], "term": r["term"], "gap": r["gap"],
            "mixed_rank": r["mixed_rank"], "aonly_rank": r["aonly_rank"],
        } for r in pushed_out],
    }
    with open(ANALYSIS_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n→ 保存: {ANALYSIS_OUT}")


if __name__ == "__main__":
    main()
