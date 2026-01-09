## About
This repository contains an educational project for the topic:

**Research and experimental comparison of machine learning testing frameworks.**

The project compares four frameworks on the same synthetic time-series forecasting task (Prophet) and the same set of data quality / drift / anomaly scenarios:
- Great Expectations
- Evidently
- Alibi Detect
- NannyML

The project is designed as a small console application that:
- Generates deterministic datasets and “scenario” variations (ideal / missing+duplicates / drift+outliers)
- Trains a forecasting model (Prophet) and evaluates prediction quality
- Runs checks from each framework and saves artifacts (HTML dashboards, JSON summaries)
- Produces a single comparison table across frameworks (time, memory, issues found, coverage, etc.)

For quick start commands (English only), see `COMMANDS.md`.

## Authors and contributors
- Alexander N. Orzhekhovskiy — main author, implementation and experiments
- Vladimir A. Parkhomenko — advisor and contributor (Senior Lecturer)

## Warranty
The contributors give **no warranty** for using this software.

## Licence
- This program is open to use anywhere and is licensed under the GNU General Public License v3.0.

## Project structure
- `main.py`: CLI entry point (runs experiments and saves reports)
- `src/`: source code
  - `frameworks/`: adapters for each framework (Great Expectations / Evidently / Alibi Detect / NannyML)
  - `model.py`: Prophet wrapper (train / forecast / metrics)
  - `reporting.py`: report generation (dashboard + summary files)
  - `data_gen.py`: baseline data generator
- `generate_scenarios.py`: scenario generator (ideal / pass / drift)
- `data/`: generated input datasets (`.csv` and `.json`)
- `reports/`: experiment artifacts (HTML dashboards, JSON summaries)

