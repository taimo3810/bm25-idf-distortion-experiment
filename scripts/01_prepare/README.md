# 01_prepare — 事前準備・調査

データ取得 → 候補単語抽出 → LLM クエリ生成までを担う。本リポでは候補単語と生成済みクエリを `data/` に同梱しているので、再現だけなら `fetch_*` 2 本だけ走らせればよい。

## 実行順

1. `fetch_and_parse_pdfs.py` — Allganize 5 業種 PDF を `pdfs/` に download し、page-level で `chunks.jsonl` 出力。`data/documents.csv` を入力に使う。
2. `fetch_jagovfaqs.py` — `sbintuitions/JMTEB` から JaGovFaqs-22k を fetch して `jagov_chunks.jsonl` 出力。
3. (任意) `select_direction_a_terms.py` / `select_direction_b_terms.py` — Sudachi で形態素解析し、過小評価方向 / 過大評価方向の候補単語を抽出して `data/direction_*_candidates.json` に出力。出力済みのものを上書きする。
4. (任意) `generate_queries.py` / `generate_queries_dir_b.py` — `gpt-4o-mini` に候補単語+chunk_text を渡してサポートデスク問い合わせ風クエリを生成。`data/generated_queries*.json` に出力。OpenAI API キーが必要。
