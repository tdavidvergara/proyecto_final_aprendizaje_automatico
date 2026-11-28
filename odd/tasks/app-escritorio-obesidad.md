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
- [x] T3 — `pyproject.toml` (`[project.scripts] app = ...`), `uv run app` works; README section "App de escritorio".

- [x] T4 — Integrate `origin/develop` commit `459b490` (Sebastián, 2026-11-28: retrained `modelo_ganador.pkl` pickled with P2 functions in `__main__`, `docs/GUIA_DESPLIEGUE_INTERFAZ.md`): merge into this branch, resolve the `README.md` conflict keeping both sections, adapt `core.py` to register `agregar_habitos` / `nombres_con_habitos` in `__main__` before unpickling (per the guide), keep the Windows booster fallback only if still needed, tests green. Reason: the new pickle fails to load (`Can't get attribute 'agregar_habitos' on <module '__main__'>`, 3 tests fail).
- [ ] T5 — Review follow-ups that harden the model swap: golden prediction test from an independent source (notebook output) if available, direct test for the booster fallback if it remains, GUI shows a readable error when the model cannot load.
- [ ] T6 — Delivery: push `feat/app-escritorio` and open a PR into `develop` (user delegated push/PR to the agent on 2026-11-28).

## Acceptance criteria

- `uv run pytest` passes (existing + new tests).
- `uv run app` opens the window; a prediction for a sample profile shows a class and 7 probabilities summing to ~100%.

## Route

- T1–T3 delegated to one writer (2+ non-trivial files; reading the pickle/notebook prepares the write).

## Delivery

- Forecast ~450 authored lines. Strategy: ask-on-risk; no PR is being created now, so the chain question is deferred to PR time.
- 2026-11-28: user delegated push/PR. Strategy `exception-ok`, single PR into `develop`: the app is one coherent unit (~700 authored lines; `uv.lock` is generated), and reviewers are classmates, not a chained-review team.

## Open notes

- The pickles were written on Linux (Colab) and fail to unpickle on Windows with XGBoost 3.4.1 ("input stream corrupted"): the booster Config section stores the libstdc++ mt19937 RNG state (625 decimal tokens) while MSVC expects 624 hex tokens. `core.load_model` rebuilds the booster from the portable "Model" section only (prediction does not use the RNG). No change to `modelos/`.
- `dataset/ObesityDataSet_raw_and_data_sinthetic.csv` has European-formatted numbers (e.g. `2.450.218`), so the notebook test split (which used `fetch_ucirepo`) cannot be reproduced offline from it. Sanity check on the 489 cleanly parseable unique rows: accuracy 0.896 (mostly training rows).

- Winner model uses `Gender` and `Age`; an earlier team decision excluded them from the preventive benchmark. Flagged to the user; not changed here.

## Progress

- 2026-11-28: feature document created; branch created.
- 2026-11-28: T1 done (delegated writer). RED observed (ImportError: cannot import `core`), then GREEN: `uv run pytest -q` -> 29 passed. Commit `1e7385f`. Out-of-range policy: reject with ValueError.
- 2026-11-28: T2 done (delegated writer). GUI smoke: window built with title "Clasificador de Obesidad", classify -> "Sobrepeso nivel I" with probabilities summing to 1.0, reset restores defaults, window destroyed after 1 s via `after()`. Test-first exception: GUI logic is thin and covered by the smoke run; model logic is tested in T1. Commit `06a449d`.
- 2026-11-28: T3 done (delegated writer). `pyproject.toml` (hatchling, packages `app_obesidad` + `src`, `app = "app_obesidad.app:main"`), `uv.lock`, README "App de escritorio". Pins: scikit-learn 1.6.1 and xgboost 3.4.1 from Notebook_Matriz.ipynb cell 3 output; pickle loads with no InconsistentVersionWarning. Checks: `uv run pytest -q` -> 29 passed; `core.predict` sample -> Normal_Weight, 7 probs sum 1.0; GUI import ok; window built and destroyed via `after(1000)`. `app.exe` entry point present in `.venv/Scripts`; interactive `uv run app` not launched (would block). Commit `0c9fa3e`.
- 2026-11-28: Native review (base `develop`, 8 files, risk medium, consent granted, lens reliability): approved and acknowledged. Non-blocking follow-ups: golden prediction test against a known profile; direct test for the Windows booster fallback; catch model-load errors in the GUI; test GUI vocabulary mapping; tighten pandas/numpy version bounds; validate range before rounding.
- 2026-11-28: Second candidate (base merge-base with `main`, 47 files / ~267k lines incl. existing team work) granted but stopped with `lens_context_budget_exceeded`; no authority created.
- 2026-11-28: T4 done (delegated writer). Merged `origin/develop` (`65a19f6`, README conflict resolved keeping the team's "Entrega Final" section and the "App de escritorio" section). RED: `uv run pytest -q` -> 3 failed (`Can't get attribute 'agregar_habitos' on <module '__main__'>`). Pickle opcodes reference `__main__.agregar_habitos` and `__main__.nombres_con_habitos`; `core.register_p2_on_main()` sets them from `src.p2_habitos` (`agregar_habitos`, `_nombres_con_habitos`) only when missing. Booster fallback still required: plain `joblib.load` with the callables registered fails with `XGBoostError: input stream corrupted`. GREEN: 30 passed. GUI smoke: sidebar shows `M2_XGBoost`, `P2_Habitos_YeoJohnson`, 79.67%, 0.7939, `2026-11-28 07:17`; window destroyed via `after(1000)`. Commit `ac85234`.
- Next: user runs `uv run app` interactively; decide on PR / delivery, the Gender/Age open note, and whether to address the review follow-ups.
