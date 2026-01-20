# AI-Detector

FraudSquad++ experiment scaffold

Quick start (inspect -> graph -> splits):
1) Inspect and standardize nodes (auto timestamp)
   python -m src.preprocess.inspect_and_standardize \
     --in_path /abs/path/to/llama3.jsonl \
     --out_dir /root/autodl-fs/fraudsquad_project/artifacts/graphs/Amazon-Llama3

2) Build graph + nodes table
   python -m src.preprocess.build_graph_dgl \
     --dataset Amazon-Llama3 \
     --in /root/autodl-fs/fraudsquad_project/artifacts/graphs/Amazon-Llama3/nodes.parquet \
     --format parquet \
     --out /root/autodl-fs/fraudsquad_project/artifacts/graphs/Amazon-Llama3

3) Create splits
   python -m src.preprocess.make_splits \
     --dataset Amazon-Llama3 \
     --nodes /root/autodl-fs/fraudsquad_project/artifacts/graphs/Amazon-Llama3/nodes.parquet \
     --out /root/autodl-fs/fraudsquad_project/artifacts/splits/Amazon-Llama3 \
     --seeds 2024 2025 2026 \
     --train_ratio 0.01 --val_ratio 0.09 --test_ratio 0.90
