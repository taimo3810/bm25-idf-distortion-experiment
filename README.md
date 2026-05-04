# bm25-idf-distortion-experiment

Elasticsearch で複数テナントを 1 つの index にまとめて検索時にメタデータでフィルターすると、BM25 の IDF が全コーパス統計から計算されたまま使われ、検索ランキングが歪む — その現象を実データで再現する実験。

📝 **ブログ記事**: https://zenn.dev/taimo/articles/e354be5e920bc9

## 結論

異業種マルチテナント (Allganize 5 業種 + JaGovFaqs 22.7K = 24.2K チャンク) を 1 index に同居させ、サポートデスク問い合わせ風クエリ 128 件 (過小評価方向) / 44 件 (過大評価方向) で測定した結果:

| 観点 | 過小評価方向 (実験 1) | 過大評価方向 (実験 2) |
|---|---|---|
| 平均 NDCG@5 差 (分離 − 共有) | **+0.080** | +0.043 |
| 平均 Recall@5 差 (分離 − 共有) | +0.047 | **+0.068** |
| Top-5 押出クエリ比率 | 4.7% (6/128) | **6.8% (3/44)** |

→ 共有 index ではテナント横断の単語頻度が IDF に持ち込まれ、テナント内では珍しい/ありふれた語の重みが歪む。RAG で Top-5 を LLM コンテキストに詰める設計だと、正解チャンクが弾かれて回答品質が落ちる。

## 環境前提

- **Python** 3.11+
- **Elasticsearch** 9.2+ (Elastic Cloud Serverless または self-hosted)
  - 必須プラグイン: `analysis-kuromoji`
    - Serverless: 標準同梱、追加作業不要
    - self-hosted: `bin/elasticsearch-plugin install analysis-kuromoji`
- **Sudachi 辞書バージョン** `sudachidict-core==20260116` で実験。バージョンが変わると候補単語抽出結果が変わるので注意。
- **(任意) OpenAI API key**: クエリを再生成する場合のみ必要。本リポは生成済みクエリ (`data/generated_queries*.json`) を同梱しているので、検証だけなら不要。

## Quickstart

### 1. インストール

```bash
git clone <this-repo>
cd bm25-idf-distortion-experiment
uv sync   # または pip install -e .
```

### 2. `.env` 設定

```bash
cp .env.example .env
# 値を埋める: ELASTICSEARCH_URL, ELASTICSEARCH_API_KEY (再現だけなら OPENAI_API_KEY は不要)
```

### 3. データ取得 (1 回だけ)

```bash
python scripts/01_prepare/fetch_and_parse_pdfs.py   # → pdfs/ + chunks.jsonl
python scripts/01_prepare/fetch_jagovfaqs.py        # → jagov_chunks.jsonl
```

### 4. 検証実行

```bash
python scripts/02_experiment/verify_multi_tenant_v2.py     # 過小評価方向 (記事 §実験 1)
python scripts/02_experiment/verify_multi_tenant_dir_b.py  # 過大評価方向 (記事 §実験 2)
python scripts/03_analyze/analyze_k5.py                    # K=5 集計 (記事の数値)
```

### (任意) クエリをフル再生成する場合 (OpenAI key 必要)

`data/` 同梱版で十分なら不要。再生成したい場合:

```bash
python scripts/01_prepare/select_direction_a_terms.py
python scripts/01_prepare/select_direction_b_terms.py
python scripts/01_prepare/generate_queries.py
python scripts/01_prepare/generate_queries_dir_b.py
```

## ディレクトリ構成

```
bm25-idf-distortion-experiment/
├── scripts/
│   ├── 01_prepare/      # データ取得 → 候補単語抽出 → クエリ生成
│   ├── 02_experiment/   # mixed vs aonly 検証 (dfs_query_then_fetch)
│   └── 03_analyze/      # K=5 集計、押出率計算
├── data/                # 中間生成物 (commit 済み、ライセンスは THIRD_PARTY_NOTICES.md 参照)
│   ├── documents.csv                  # Allganize PDF メタ
│   ├── direction_a_candidates.json    # 過小評価候補単語 + chunk_text 抜粋
│   ├── direction_b_candidates.json    # 過大評価候補単語 + chunk_text 抜粋
│   ├── generated_queries.json         # gpt-4o-mini 生成クエリ (実験 1 用)
│   └── generated_queries_dir_b.json   # 同 (実験 2 用)
└── results/             # 実験結果
    ├── results_multi_tenant_v2.json    # NDCG@10 / Recall@50 / MRR / RBO@50
    ├── results_multi_tenant_dir_b.json # 同 (過大評価方向)
    └── analysis_k5.json                # K=5 集計 (記事の数値)
```

## 実装メモ

- **Elasticsearch インデックススキーマ**: `kuromoji_tokenizer` + `kuromoji_baseform` + `lowercase` フィルタ。BM25 デフォルトパラメータ (`k1=1.2`, `b=0.75`)。
- **検索 search_type**: `dfs_query_then_fetch` でシャード横断 IDF を集約。それでも shard 数や routing で結果は微差する。
- **シャード設定**: `INDEX_BODY` には `number_of_shards` を未指定 (Serverless の auto-shard を許容)。記事数値と完全一致させたい場合は `verify_multi_tenant_v2.py` / `_dir_b.py` の `INDEX_BODY` の `settings` 配下に `"index": {"number_of_shards": 1}` を追加して self-hosted で動かす。
- **K=5 への再集計**: `verify_*.py` は K=10 / K=50 で出力する。Top-5 押出率など記事の数値は `analyze_k5.py` が再計算したもの。

## 候補単語抽出の閾値 (記事には書いていない実装詳細)

`select_direction_*_terms.py` の出力条件:

- 過小評価方向 (Direction A): テナント内出現率 < 全体出現率、出現率比 (全体/テナント内) ≥ 4、テナント内最低出現数 5 件以上
- 過大評価方向 (Direction B): テナント内出現率 > 全体出現率、出現率比 (テナント内/全体) ≥ 50、テナント内最低出現数 10 件以上
- 形態素解析: Sudachi `SplitMode.C`、品詞は名詞のみ抽出 (動詞・助詞等は除外)

詳細はスクリプト本体の閾値定数を参照。

## ライセンス

- ソースコード: MIT (`LICENSE` 参照)
- `data/` 配下の派生データ: 元データのライセンスを継承 (`THIRD_PARTY_NOTICES.md` 参照)
  - Allganize 由来分: MIT
  - JaGovFaqs (sbintuitions/JMTEB) 由来分: CC-BY-4.0

## 参考

- Elastic 公式: [Getting consistent scoring](https://www.elastic.co/docs/solutions/search/full-text/search-relevance/consistent-scoring)
- Elastic 公式ブログ: [Understanding Query Then Fetch vs DFS Query Then Fetch](https://www.elastic.co/blog/understanding-query-then-fetch-vs-dfs-query-then-fetch)
- Allganize/RAG-Evaluation-Dataset-JA: https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-JA
- sbintuitions/JMTEB (JaGovFaqs-22k): https://huggingface.co/datasets/sbintuitions/JMTEB
