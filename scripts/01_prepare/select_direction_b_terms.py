"""Direction B (過大評価) 候補単語の精査.

Direction A の逆: テナント内で common (= 業種特有語) ∩ 全体で rare な単語。
共有 index ではこの単語の IDF が高くなり、ノイズ chunk のスコアも底上げされる。

設計:
  - 各業種テナント内で:
    * subset_count: ≥ 25 (= テナント内で頻出, 業種特有)
    * subset_ratio: ≥ 0.1 (= 10% 以上の chunk に出現)
    * global_count: ≤ subset_count × 2 (= テナント外にはほぼ出ない)
    * 過大評価方向の gap: subset_ratio / global_ratio ≥ 4
    * stopword 除外
  - 各候補単語に「target chunk 1 件」を紐付ける
    + さらに「同テナント内で同じ単語を含む他 chunks (= 潜在ノイズ chunks)」が複数存在することを確認

Output:
  - direction_b_candidates.json: [{"tenant", "term", "chunk_doc_id", "chunk_text", "n_other_chunks_with_term", "stats"}, ...]
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from sudachipy import Dictionary, SplitMode

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ALLGANIZE_CHUNKS = REPO_ROOT / "chunks.jsonl"
JAGOV_CHUNKS = REPO_ROOT / "jagov_chunks.jsonl"
CANDIDATES_OUT = DATA_DIR / "direction_b_candidates.json"

ALLGANIZE_TENANTS = ["finance", "it", "manufacturing", "public", "retail"]
NOISE_TENANT = "jagov_general"

STOPWORDS = {
    "ください", "下さい", "致します", "いたします", "ござい", "ある", "あり", "あれ",
    "いる", "おる", "なる", "する", "やる", "できる", "おく", "いく", "くる", "みる",
    "しまう", "ござる", "持つ", "受ける", "行う", "示す", "見る",
    "こと", "もの", "とき", "ため", "ところ", "ほう", "よう", "ふう", "うち", "そば",
    "場合", "中", "上", "下", "左", "右", "前", "後", "間", "側", "外", "内", "他",
    "者", "方", "者等", "等", "自身", "本人", "全体", "部分", "通り",
    "以上", "以下", "以内", "以外", "以前", "以後", "未満", "限り", "多く", "少し",
    "全て", "すべて", "全部", "一部", "一つ", "二つ", "三つ", "何", "誰", "何人",
    "千", "百", "十", "万", "億", "兆", "数",
    "実施", "対応", "対象", "目的", "内容", "方法", "結果", "影響", "状況", "状態",
    "理由", "種類", "観点", "範囲", "規定", "詳細", "概要", "事項", "事例", "情報",
    "問題", "課題", "意見", "意向", "予定", "可能", "必要", "重要", "中心", "基本",
    "可能性", "重要性", "必要性",
    "国内", "国外", "地域", "場所", "本部", "本社", "支社", "施設", "現場",
    "今後", "今回", "今年", "今月", "今日", "今度", "現在", "最近", "最新",
    "特に", "また", "なお", "ただし", "そして", "しかし", "さらに", "つまり", "例えば",
    "及び", "または", "もしくは", "かつ", "ない", "良い",
    "教え", "答え", "違い", "差異", "関係", "関連", "提供", "利用", "活用", "確認",
    "実例", "比較", "判断", "措置", "改善", "向上", "増加", "減少",
}

HIRA_ONLY = re.compile(r"^[぀-ゟ]+$")
PUNCT_OR_DIGIT = re.compile(r"^[\d\W_]+$")


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


def is_meaningful(surf: str) -> bool:
    if surf in STOPWORDS:
        return False
    if len(surf) < 2:
        return False
    if PUNCT_OR_DIGIT.match(surf):
        return False
    if HIRA_ONLY.match(surf) and len(surf) <= 3:
        return False
    return True


def main() -> None:
    print("=" * 70)
    print("Direction B (過大評価) 候補単語の精査")
    print("=" * 70)

    print("\n[1/4] chunks 読み込み...")
    chunks = load_chunks()
    print(f"  total: {len(chunks):,}")

    print("\n[2/4] tokenize + df 計算 ...")
    tokenizer = Dictionary().create()
    global_df: Counter = Counter()
    tenant_df: dict[str, Counter] = defaultdict(Counter)
    tenant_n: dict[str, int] = defaultdict(int)
    tenant_chunks_for_term: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for i, c in enumerate(chunks):
        if i % 5000 == 0 and i > 0:
            print(f"    {i:,}/{len(chunks):,}...")
        terms = set()
        for m in tokenizer.tokenize(c["text"], SplitMode.A):
            pos = m.part_of_speech()[0]
            if pos != "名詞":
                continue
            surf = m.surface()
            if not is_meaningful(surf):
                continue
            terms.add(surf)
        tenant = c["tenant"]
        tenant_n[tenant] += 1
        for t in terms:
            global_df[t] += 1
            tenant_df[tenant][t] += 1
            if tenant in ALLGANIZE_TENANTS:
                tenant_chunks_for_term[t][tenant].append(c)

    n_global = len(chunks)
    print(f"  vocab: {len(global_df):,}")
    for t, n in sorted(tenant_n.items()):
        print(f"    {t}: {n:,}")

    print("\n[3/4] 各業種テナントで Direction B 候補単語を抽出 ...")
    candidates: list[dict] = []
    for tenant in ALLGANIZE_TENANTS:
        tn = tenant_n[tenant]
        scored = []
        for term, sub_n in tenant_df[tenant].items():
            if sub_n < 25:  # テナント内で頻出
                continue
            sub_ratio = sub_n / tn
            if sub_ratio < 0.1:
                continue
            gn = global_df[term]
            if gn > sub_n * 2:  # テナント外にはほぼ出ない (= 業種特有)
                continue
            global_ratio = gn / n_global
            if global_ratio == 0:
                gap = float("inf")
            else:
                gap = sub_ratio / global_ratio
            if gap < 4:
                continue
            scored.append((term, sub_n, gn, sub_ratio, global_ratio, gap))
        scored.sort(key=lambda x: -x[5])  # gap 大きい順
        print(f"  [{tenant}] candidates: {len(scored)}")
        for term, sub_n, gn, sub_r, gl_r, gap in scored[:25]:
            chunk_pool = tenant_chunks_for_term[term][tenant]
            n_other = len(chunk_pool)
            if n_other < 5:  # 同単語含む chunk が複数あること = ノイズ chunks 候補確保
                continue
            print(f"    {term}: sub_n={sub_n}, global_n={gn}, gap={gap:.1f}, n_other_chunks={n_other}")
            chunk_pool_sorted = sorted(chunk_pool, key=lambda c: abs(len(c["text"]) - 600))
            chunk = chunk_pool_sorted[0]
            candidates.append({
                "tenant": tenant,
                "term": term,
                "subset_count": sub_n,
                "subset_ratio": round(sub_r, 5),
                "global_count": gn,
                "global_ratio": round(gl_r, 5),
                "gap": round(gap, 2),
                "n_other_chunks_with_term": n_other,
                "chunk_doc_id": chunk["doc_id"],
                "chunk_file_name": chunk["file_name"],
                "chunk_text": chunk["text"][:1500],
            })

    print(f"\n  total candidates: {len(candidates)}")

    print("\n[4/4] 出力 ...")
    with open(CANDIDATES_OUT, "w", encoding="utf-8") as f:
        json.dump({"n_candidates": len(candidates), "candidates": candidates}, f, ensure_ascii=False, indent=2)
    print(f"  → {CANDIDATES_OUT}")


if __name__ == "__main__":
    main()
