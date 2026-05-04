# 03_analyze — 事後分析

`02_experiment/verify_*` の結果 (K=10/50 ベース) を K=5 に絞り直し、押出率・順位低下率を計算する。記事に載っている数値はここから出ている。

## スクリプト

- `analyze_k5.py` — `results/results_multi_tenant_v2.json` を読み、K=5 メトリクス (NDCG@5, Recall@5, Top-5 押出率) を計算して `results/analysis_k5.json` に出力。

## なぜ K=5 に絞り直すか

verify は K=10/50 で多様な指標を出すが、RAG では LLM コンテキストに詰めるチャンク数として K=5 がよく使われる。K=5 で「Top-5 の正解チャンクが共有 index 構成で押し出される確率」を計算するのが、本実験の主結論。
