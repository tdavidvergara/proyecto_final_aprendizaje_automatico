# Feature: app-escritorio-obesidad

Locator: `odd/tasks/app-escritorio-obesidad.md` · Engram mirror: `odd/app-escritorio-obesidad/tasks` · Branch: `feat/app-escritorio` (from `develop`)

## Objective

Deploy the winning model (`modelos/modelo_ganador.pkl`, XGBoost M2 + P2 habits/Yeo-Johnson, 7 obesity classes) locally as a Windows desktop app, following the pattern shown in class (PyCaret Iris example): `core.py` loads the model and predicts, `app.py` is a CustomTkinter GUI, launched with `uv run app`.

## Problem / Why

The team needs a usable front for the final project. The repo already exports the winner model and `metricas_ganador.json` (columns, classes, ranges) "ready for the deployment app", but no app exists.

## Scope

- In: `app_obesidad/` package (core + GUI), tests for core, `pyproject.toml` for `uv run app`, README usage section.
- Out: PyInstaller `.exe` and Inno Setup installer (optional later), retraining or changing the model, report/notebook edits.

## Constraints

- The pickle depends on `src/p2_habitos.py` (`agregar_habitos`); it must stay importable as it was when pickled.
- Library versions must be compatible with the pickled pipeline (notebook kernel: Python 3.13.1).
- English for code, comments and identifiers; UI copy in Spanish (end users are Spanish speakers, as in the class example).
- No push / PR without the user's decision.

## Tasks

- [x] T1 — `app_obesidad/core.py`: load model + metrics JSON, map friendly inputs to the model's 17 columns, return predicted class + per-class probabilities. Tests in `tests/test_core.py` (RED → GREEN).
- [x] T2 — `app_obesidad/app.py`: CustomTkinter GUI with inputs driven by `metricas_ganador.json` ranges, Classify/Reset, probability bars for the 7 classes, model info sidebar.
- [ ] T3 — `pyproject.toml` (`[project.scripts] app = ...`), `uv run app` works; README section "App de escritorio".

## Acceptance criteria

- `uv run pytest` passes (existing + new tests).
- `uv run app` opens the window; a prediction for a sample profile shows a class and 7 probabilities summing to ~100%.

## Route

- T1–T3 delegated to one writer (2+ non-trivial files; reading the pickle/notebook prepares the write).

## Delivery

- Forecast ~450 authored lines. Strategy: ask-on-risk; no PR is being created now, so the chain question is deferred to PR time.

## Open notes

- The pickles were written on Linux (Colab) and fail to unpickle on Windows with XGBoost 3.4.1 ("input stream corrupted"): the booster Config section stores the libstdc++ mt19937 RNG state (625 decimal tokens) while MSVC expects 624 hex tokens. `core.load_model` rebuilds the booster from the portable "Model" section only (prediction does not use the RNG). No change to `modelos/`.
- `dataset/ObesityDataSet_raw_and_data_sinthetic.csv` has European-formatted numbers (e.g. `2.450.218`), so the notebook test split (which used `fetch_ucirepo`) cannot be reproduced offline from it. Sanity check on the 489 cleanly parseable unique rows: accuracy 0.896 (mostly training rows).

- Winner model uses `Gender` and `Age`; an earlier team decision excluded them from the preventive benchmark. Flagged to the user; not changed here.

## Progress

- 2026-11-28: feature document created; branch created.
- 2026-11-28: T1 done (delegated writer). RED observed (ImportError: cannot import `core`), then GREEN: `uv run pytest -q` -> 29 passed. Commit `1e7385f`. Out-of-range policy: reject with ValueError.
- 2026-11-28: T2 done (delegated writer). GUI smoke: window built with title "Clasificador de Obesidad", classify -> "Sobrepeso nivel I" with probabilities summing to 1.0, reset restores defaults, window destroyed after 1 s via `after()`. Test-first exception: GUI logic is thin and covered by the smoke run; model logic is tested in T1. Commit `06a449d`.
