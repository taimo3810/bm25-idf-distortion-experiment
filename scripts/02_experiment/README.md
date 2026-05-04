# 02_experiment — 実験本体

共有インデックス (mixed) と 分離インデックス (aonly) を作成し、同じクエリを両方に投げて検索結果を比較する。内部で `dfs_query_then_fetch` を使ってシャード横断 IDF を集約している。

## スクリプト

- `verify_multi_tenant_v2.py` — 過小評価方向 (記事 §実験 1)。`data/generated_queries.json` を入力、`results/results_multi_tenant_v2.json` を出力 (NDCG@10 / Recall@50 / MRR / RBO@50)。
- `verify_multi_tenant_dir_b.py` — 過大評価方向 (記事 §実験 2)。`data/generated_queries_dir_b.json` を入力、`results/results_multi_tenant_dir_b.json` を出力。

## 前提

- `chunks.jsonl` (Allganize) と `jagov_chunks.jsonl` (JaGovFaqs) が `01_prepare/fetch_*` で生成済み
- `.env` に `ELASTICSEARCH_URL` / `ELASTICSEARCH_API_KEY` が設定済み
- Elasticsearch 9.2+ (`analysis-kuromoji` プラグイン同梱) が起動中
