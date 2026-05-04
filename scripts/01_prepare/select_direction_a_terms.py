"""各業種テナント内で「Direction A 候補単語」を精査して chunk 紐付き候補を出力.

設計:
  - Allganize 5 業種 + JaGovFAQs ノイズプール の構成で
  - 各業種テナント内で:
    * subset_count: 1 〜 10 件 (= テナント内に存在するが rare)
    * global_count: ≥ 500 (= 全体では common)
    * gap (global_ratio / subset_ratio): ≥ 4
    * stopword 除外
  - 各候補単語に「該当 chunk」を 1 件紐付ける (= LLM で query 生成する材料)

Output:
  - direction_a_candidates.json: [{"tenant", "term", "chunk_doc_id", "chunk_text", "stats"}, ...]
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
CANDIDATES_OUT = DATA_DIR / "direction_a_candidates.json"

ALLGANIZE_TENANTS = ["finance", "it", "manufacturing", "public", "retail"]
NOISE_TENANT = "jagov_general"

# 拡張 stopword: 助動詞・指示語・形式名詞・一般動詞・補助動詞・数量詞・典型副詞
STOPWORDS = {
    # 助動詞・補助動詞・形式語
    "ください", "下さい", "致します", "いたします", "ござい", "ある", "あり", "あれ",
    "いる", "おる", "なる", "する", "やる", "できる", "おく", "いく", "くる", "みる",
    "しまう", "ござる", "持つ", "受ける", "行う", "示す", "見る",
    # 形式名詞・一般名詞
    "こと", "もの", "とき", "ため", "ところ", "ほう", "よう", "ふう", "うち", "そば",
    "場合", "中", "上", "下", "左", "右", "前", "後", "間", "側", "外", "内", "他",
    "者", "方", "者等", "等", "自身", "本人", "全体", "部分", "通り",
    # 程度・数量・指示
    "以上", "以下", "以内", "以外", "以前", "以後", "未満", "限り", "多く", "少し",
    "全て", "すべて", "全部", "一部", "一つ", "二つ", "三つ", "何", "誰", "何人",
    "千", "百", "十", "万", "億", "兆", "数",
    # 抽象動作名詞 (情報量が低くノイズになりやすい)
    "実施", "対応", "対象", "目的", "内容", "方法", "結果", "影響", "状況", "状態",
    "理由", "種類", "観点", "範囲", "規定", "詳細", "概要", "事項", "事例", "情報",
    "問題", "課題", "意見", "意向", "予定", "可能", "必要", "重要", "中心", "基本",
    "可能性", "重要性", "必要性",
    # 場所・組織のメタ語
    "国内", "国外", "地域", "場所", "本部", "本社", "支社", "施設", "現場",
    # 時間
    "今後", "今回", "今年", "今月", "今日", "今度", "現在", "最近", "最新",
    # 副詞
    "特に", "また", "なお", "ただし", "そして", "しかし", "さらに", "つまり", "例えば",
    "及び", "または", "もしくは", "かつ", "ない", "良い",
    # その他高頻度ノイズ
    "教え", "答え", "違い", "差異", "関係", "関連", "提供", "利用", "活用", "確認",
    "実例", "比較", "判断", "措置", "改善", "向上", "増加", "減少",
}

# 単語長 / 文字種フィルタ用 regex
HIRA_ONLY = re.compile(r"^[぀-ゟ]+$")
KATA_ONLY = re.compile(r"^[゠-ヿー]+$")
PUNCT_OR_DIGIT = re.compile(r"^[\d\W_]+$")


def load_chunks() -> list[dict]:
    chunks: list[dict] = []
    with open(ALLGANIZE_CHUNKS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            chunks.append({
                "doc_id": r["doc_id"],
                "tenant": r["domain"],
                "file_name": r["file_name"],
                "page_no": r.get("page_no"),
                "text": r["text"],
            })
    with open(JAGOV_CHUNKS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            chunks.append({
                "doc_id": r["doc_id"],
                "tenant": NOISE_TENANT,
                "file_name": r["doc_id"],
                "page_no": None,
                "text": r["text"],
            })
    return chunks


def is_meaningful(surf: str) -> bool:
    """意味のある単語かどうか (stopword 除外 + 形式フィルタ)."""
    if surf in STOPWORDS:
        return False
    if len(surf) < 2:
        return False
    if PUNCT_OR_DIGIT.match(surf):
        return False
    if HIRA_ONLY.match(surf) and len(surf) <= 3:
        return False  # 平仮名のみ短語は除外 ("する", "ある" など)
    return True


def main() -> None:
    print("=" * 70)
    print("Direction A 候補単語の精査")
    print("=" * 70)

    print("\n[1/4] chunks 読み込み...")
    chunks = load_chunks()
    print(f"  total: {len(chunks):,}")

    print("\n[2/4] tokenize + df 計算 ...")
    tokenizer = Dictionary().create()
    global_df: Counter = Counter()
    tenant_df: dict[str, Counter] = defaultdict(Counter)
    tenant_n: dict[str, int] = defaultdict(int)
    # term → tenant → 該当 chunks (リスト)
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

    print("\n[3/4] 各業種テナントで Direction A 候補単語を抽出 ...")
    candidates: list[dict] = []
    for tenant in ALLGANIZE_TENANTS:
        tn = tenant_n[tenant]
        scored = []
        for term, sub_n in tenant_df[tenant].items():
            if not (1 <= sub_n <= 10):  # テナント内に存在するが rare
                continue
            gn = global_df[term]
            if gn < 500:  # 全体で common (= IDF 歪み発生条件)
                continue
            sub_ratio = sub_n / tn
            global_ratio = gn / n_global
            gap = global_ratio / sub_ratio if sub_ratio > 0 else float("inf")
            if gap < 4:
                continue
            scored.append((term, sub_n, gn, sub_ratio, global_ratio, gap))
        scored.sort(key=lambda x: -x[5])  # gap 大きい順
        print(f"  [{tenant}] candidates: {len(scored)}")
        for term, sub_n, gn, sub_r, gl_r, gap in scored[:25]:
            print(f"    {term}: sub_n={sub_n}, global_n={gn}, gap={gap:.1f}")
            # 該当 chunk を 1 件選ぶ (= LLM 生成材料)
            chunk_pool = tenant_chunks_for_term[term][tenant]
            if not chunk_pool:
                continue
            # 短すぎず長すぎない chunk を優先
            chunk_pool.sort(key=lambda c: abs(len(c["text"]) - 600))
            chunk = chunk_pool[0]
            candidates.append({
                "tenant": tenant,
                "term": term,
                "subset_count": sub_n,
                "subset_ratio": round(sub_r, 5),
                "global_count": gn,
                "global_ratio": round(gl_r, 5),
                "gap": round(gap, 2),
                "chunk_doc_id": chunk["doc_id"],
                "chunk_file_name": chunk["file_name"],
                "chunk_text": chunk["text"][:1500],  # LLM プロンプト用に短縮
            })

    print(f"\n  total candidates: {len(candidates)}")

    print("\n[4/4] 出力 ...")
    with open(CANDIDATES_OUT, "w", encoding="utf-8") as f:
        json.dump({"n_candidates": len(candidates), "candidates": candidates}, f, ensure_ascii=False, indent=2)
    print(f"  → {CANDIDATES_OUT}")


if __name__ == "__main__":
    main()
