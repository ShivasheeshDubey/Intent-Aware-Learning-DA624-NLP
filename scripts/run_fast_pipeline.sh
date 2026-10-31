#!/usr/bin/env bash
set -e

MODEL="Qwen/Qwen3-1.7B-Base"
FILTER="Qwen/Qwen3-1.7B"
mkdir -p data/processed data/splits checkpoints results

python scripts/prepare_data.py --dataset c4 --model "$MODEL" --max-samples 120 --output data/processed/c4_content.jsonl
python scripts/generate_candidates.py --input data/processed/c4_content.jsonl --output data/processed/c4_candidates.jsonl --model "$MODEL" --max-targets-per-sequence 4 --top-k 100 --max-candidates 30
python scripts/filter_concepts.py --input data/processed/c4_candidates.jsonl --output data/processed/c4_concepts.jsonl --model "$FILTER" --batch-size 8 --max-synonyms 10
python scripts/split_data.py --input data/processed/c4_concepts.jsonl --output-dir data/splits/c4

python scripts/prepare_data.py --dataset openwebtext --model "$MODEL" --max-samples 60 --output data/processed/openwebtext_eval.jsonl

python scripts/train_model.py --train data/splits/c4/train.jsonl --output-dir checkpoints/ntp --model "$MODEL" --concept-weight 0.0 --epochs 2
python scripts/train_model.py --train data/splits/c4/train.jsonl --output-dir checkpoints/concept --model "$MODEL" --concept-weight 0.5 --target-mass 0.6 --epochs 2
python scripts/train_model.py --train data/splits/c4/train.jsonl --output-dir checkpoints/ours --model "$MODEL" --concept-weight 0.5 --target-mass 0.6 --preserve-beta 0.1 --epochs 2

python scripts/evaluate_model.py --data data/splits/c4/test.jsonl --output results/c4_base.json --base-model "$MODEL"
python scripts/evaluate_model.py --data data/splits/c4/test.jsonl --output results/c4_ntp.json --base-model "$MODEL" --adapter checkpoints/ntp
python scripts/evaluate_model.py --data data/splits/c4/test.jsonl --output results/c4_concept.json --base-model "$MODEL" --adapter checkpoints/concept
python scripts/evaluate_model.py --data data/splits/c4/test.jsonl --output results/c4_ours.json --base-model "$MODEL" --adapter checkpoints/ours

python scripts/evaluate_model.py --data data/processed/openwebtext_eval.jsonl --output results/owt_base.json --base-model "$MODEL"
python scripts/evaluate_model.py --data data/processed/openwebtext_eval.jsonl --output results/owt_concept.json --base-model "$MODEL" --adapter checkpoints/concept
python scripts/evaluate_model.py --data data/processed/openwebtext_eval.jsonl --output results/owt_ours.json --base-model "$MODEL" --adapter checkpoints/ours

python scripts/compare_results.py --base results/c4_base.json --concept results/c4_concept.json --output results/bootstrap_concept_vs_base.json
python scripts/download_benchmarks.py --output-dir data/benchmarks
python scripts/evaluate_similarity.py --benchmark-dir data/benchmarks --output results/semantic_base.json --base-model "$MODEL" --max-pairs 300
python scripts/evaluate_similarity.py --benchmark-dir data/benchmarks --output results/semantic_concept.json --base-model "$MODEL" --adapter checkpoints/concept --max-pairs 300
python scripts/evaluate_similarity.py --benchmark-dir data/benchmarks --output results/semantic_ours.json --base-model "$MODEL" --adapter checkpoints/ours --max-pairs 300
python scripts/render_report.py
