# Colab emergency runbook

1. Select a GPU runtime (T4 is sufficient for the fast configuration).
2. Clone this repository and `cd` into it.
3. Run:

```bash
pip install -q -r requirements.txt
python -m spacy download en_core_web_sm
pip install -q -e .
```

4. Execute:

```bash
bash scripts/run_fast_pipeline.sh
```

5. The important outputs are:

- `results/c4_base.json`
- `results/c4_ntp.json`
- `results/c4_concept.json`
- `results/c4_ours.json`
- `results/owt_base.json`
- `results/owt_concept.json`
- `results/owt_ours.json`
- `results/semantic_*.json`
- `results/bootstrap_concept_vs_base.json`
- `SUBMISSION_REPORT.md`

If the instruction-model filtering is too slow, reduce `--max-targets-per-sequence` from 4 to 2 in the pipeline. Do not fabricate missing results; report the reduced-scale configuration explicitly.
