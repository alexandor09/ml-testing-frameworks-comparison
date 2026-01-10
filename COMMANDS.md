# Commands (English only)

## Requirements
- Windows 10+ / macOS / Linux
- Python 3.10+

## Setup

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2) Install dependencies

```bash
python -m pip install -r requirements.txt
```

Notes (Windows / PowerShell):
- If you see `Fatal error in launcher: Unable to create process ... pip.exe`, your `.venv` was likely moved/renamed.
  Fix: recreate the venv (see below) or use `python -m pip ...` which bypasses the broken `pip.exe` launcher.

## Data generation

Generate baseline datasets:

```bash
python src/data_gen.py
```

Generate additional scenarios (ideal / missing+duplicates / drift+outliers):

```bash
python generate_scenarios.py
```

## Run

Show CLI help:

```bash
python main.py --help
```

## Google Colab

Open notebook:

```text
https://colab.research.google.com/github/alexandor09/ml-testing-frameworks-comparison/blob/main/colab_demo.ipynb
```

The notebook shows:
- where datasets are (`data/`)
- which scripts generate scenarios (`generate_scenarios.py`)
- how to run checks (`main.py`) and where "issues_detected" comes from (`src/frameworks/*_adapter.py`)
- where outputs appear (`reports/<run_name>/<timestamp>/...`)

Run all 4 frameworks (CSV):

```bash
python main.py --input data/small.csv --format csv --output reports/run_small
python main.py --input data/big.csv   --format csv --output reports/run_big
python main.py --input data/ideal.csv --format csv --output reports/run_ideal
python main.py --input data/pass.csv  --format csv --output reports/run_pass
python main.py --input data/dr.csv    --format csv --output reports/run_dr
```

## Troubleshooting (Windows)

If `python main.py ...` fails with `ModuleNotFoundError: No module named 'pandas'`, it means dependencies were not installed into the currently used interpreter.

Recommended fix (recreate venv):

```bash
deactivate
Remove-Item -Recurse -Force .venv
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Run all 4 frameworks (JSON):

```bash
python main.py --input data/small.json --format json --output reports/run_small_json
python main.py --input data/big.json   --format json --output reports/run_big_json
python main.py --input data/ideal.json --format json --output reports/run_ideal_json
python main.py --input data/pass.json  --format json --output reports/run_pass_json
python main.py --input data/dr.json    --format json --output reports/run_dr_json
```

Run a single framework:

```bash
python main.py --input data/small.csv --format csv --output reports/run_gx        --framework gx
python main.py --input data/small.csv --format csv --output reports/run_evidently --framework evidently
python main.py --input data/small.csv --format csv --output reports/run_alibi     --framework alibi
python main.py --input data/small.csv --format csv --output reports/run_nannyml   --framework nannyml
```

## Outputs

Each run creates a timestamped folder:
- `reports/<run_name>/<timestamp>/dashboard.html`
- `reports/<run_name>/<timestamp>/comparison_summary.json`
- `reports/<run_name>/<timestamp>/final_summary.json`
- `reports/<run_name>/<timestamp>/model_metrics.json`
- `reports/<run_name>/<timestamp>/<framework_name>/...`


