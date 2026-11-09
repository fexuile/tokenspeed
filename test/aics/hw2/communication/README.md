# HW2 communication experiment

This directory contains the reproducible scripts behind section 2. The
profiler writes device traces and small JSONL sidecars to an output directory;
those generated files stay outside the repository.

```bash
export PYTHON_VENV=/home/u2601213437/.venvs/tokenspeed
export AICS_MODEL=$HOME/models/Qwen3.5-9B
export AICS_OUTPUT_ROOT=/path/to/communication-output
bash test/aics/hw2/communication/run_profile.sh

python -m aics.hw2.communication.analyze \
  --root "$AICS_OUTPUT_ROOT" --output-dir "$AICS_OUTPUT_ROOT/analysis"
python -m aics.hw2.communication.plot_report \
  --analysis-dir "$AICS_OUTPUT_ROOT/analysis" \
  --output-dir docs/reports/hw2-figures
```

`intervals.py` contains the interval-union accounting used by the analysis;
`test_intervals.py` tests its boundary behavior without requiring an NPU.
