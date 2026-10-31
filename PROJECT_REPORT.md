# Project report source

The final report is generated from **actual experiment JSON files** by:

```bash
python scripts/render_report.py
```

This writes `SUBMISSION_REPORT.md` with the methodology, the real smoke-test result already obtained during development, and tables populated from whatever C4/OpenWebText/semantic evaluation files exist in `results/`.

This design prevents accidental fabrication of experiment numbers: if a run has not completed, the report displays an em dash instead of inventing a result.
