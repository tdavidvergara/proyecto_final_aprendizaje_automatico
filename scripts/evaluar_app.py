"""Evaluate the desktop app end to end on the same hold-out set as Notebook_Matriz.ipynb.

Every test row is sent through `app_obesidad.core.predict`, the exact function the GUI calls
when the user presses the predict button, so this measures the app (input encoding + model),
not only the pickled pipeline.

Split recipe copied from Notebook_Matriz.ipynb (cells 1, 2 and 16):
- UCI dataset id=544 downloaded with `ucimlrepo`.
- Exact duplicate rows removed before splitting.
- `train_test_split(test_size=0.2, stratify=y, random_state=20)`.

Usage (from the repository root):
    uv run --group eval python scripts/evaluar_app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app_obesidad import core  # noqa: E402

SEED = 20  # SEMILLA in Notebook_Matriz.ipynb
TEST_SIZE = 0.2
OUTPUT_JSON = REPO_ROOT / "modelos" / "evaluacion_app.json"
OUTPUT_PNG = REPO_ROOT / "docs" / "img" / "evaluacion_app_matriz_confusion.png"

# Example people for a live demo in class (raw survey vocabulary, as typed in the GUI).
DEMO_PROFILES = {
    "Active student, healthy habits": {
        "Gender": "Female", "Age": 21, "family_history_with_overweight": "no", "FAVC": "no",
        "FCVC": 3, "NCP": 3, "CAEC": "Sometimes", "SMOKE": "no", "CH2O": 3, "SCC": "yes",
        "FAF": 3, "TUE": 1, "CALC": "no", "MTRANS": "Walking",
    },
    "Office worker, sedentary": {
        "Gender": "Male", "Age": 35, "family_history_with_overweight": "yes", "FAVC": "yes",
        "FCVC": 2, "NCP": 3, "CAEC": "Sometimes", "SMOKE": "no", "CH2O": 2, "SCC": "no",
        "FAF": 0, "TUE": 1, "CALC": "Sometimes", "MTRANS": "Automobile",
    },
    "High-risk habits": {
        "Gender": "Female", "Age": 26, "family_history_with_overweight": "yes", "FAVC": "yes",
        "FCVC": 3, "NCP": 3, "CAEC": "Sometimes", "SMOKE": "no", "CH2O": 2, "SCC": "no",
        "FAF": 0, "TUE": 1, "CALC": "Sometimes", "MTRANS": "Public_Transportation",
    },
}


def load_hold_out() -> tuple[pd.DataFrame, pd.Series]:
    """Return the raw test rows and their true class names, split as in the notebook."""
    uci = fetch_ucirepo(id=544)
    dataset = pd.concat([uci.data.features, uci.data.targets], axis=1)
    dataset = dataset[~dataset.duplicated()]
    y = dataset["NObeyesdad"]
    # Stratify on the ordinal codes 0..6 like the notebook: stratifying on the class names
    # sorts the classes alphabetically and produces a different (and leaky-looking) split.
    levels = {name: code for code, name in enumerate(core.class_names())}
    _, idx_test = train_test_split(
        y.index, test_size=TEST_SIZE, stratify=y.map(levels), random_state=SEED)
    return dataset.loc[idx_test], y.loc[idx_test]


def predict_rows(rows: pd.DataFrame) -> tuple[list[str], list[int]]:
    """Predict every row through the app; return predictions and positions of rejected rows."""
    predictions, rejected = [], []
    for position, (_, row) in enumerate(rows.iterrows()):
        profile = {field: row[field] for field in core.INPUT_FIELDS}
        try:
            predictions.append(core.predict(profile).label)
        except ValueError as error:  # out of the training range: the GUI would refuse it too
            print(f"  rejected row {position}: {error}")
            predictions.append(None)
            rejected.append(position)
    return predictions, rejected


def save_confusion_png(matrix, names: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = ["Insuf.", "Normal", "Overw. I", "Overw. II", "Obes. I", "Obes. II", "Obes. III"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(len(names)):
        for j in range(len(names)):
            value = matrix[i, j]
            ax.text(j, i, value, ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "black")
    ax.set_xticks(range(len(names)), short, rotation=30)
    ax.set_yticks(range(len(names)), short)
    ax.set_xlabel("Predicted by the app")
    ax.set_ylabel("True class")
    ax.set_title("Desktop app on the hold-out set (XGBoost + P2)")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)


def main() -> None:
    names = core.class_names()
    reference = core.load_metadata()

    print("1) Downloading UCI dataset 544 and rebuilding the notebook split...")
    rows, y_true = load_hold_out()
    print(f"   hold-out rows: {len(rows)}")

    print("2) Predicting every row through app_obesidad.core.predict...")
    start = time.perf_counter()
    y_pred, rejected = predict_rows(rows)
    elapsed_ms = (time.perf_counter() - start) * 1000 / len(rows)

    kept = [i for i in range(len(rows)) if i not in rejected]
    true_kept = [y_true.iloc[i] for i in kept]
    pred_kept = [y_pred[i] for i in kept]

    accuracy = accuracy_score(true_kept, pred_kept)
    f1_macro = f1_score(true_kept, pred_kept, average="macro")
    matrix = confusion_matrix(true_kept, pred_kept, labels=names)

    print("\n3) Results")
    print(f"   rows evaluated : {len(kept)} (rejected: {len(rejected)})")
    print(f"   accuracy       : {accuracy:.4f}  (notebook: {reference['accuracy']:.4f})")
    print(f"   F1 macro       : {f1_macro:.4f}  (notebook: {reference['f1_macro']:.4f})")
    print(f"   time per person: {elapsed_ms:.1f} ms (encoding + prediction)\n")
    print(classification_report(true_kept, pred_kept, labels=names, digits=3))
    print(pd.DataFrame(matrix, index=names, columns=names).to_string())

    print("\n4) Demo profiles for class")
    demos = {}
    for title, profile in DEMO_PROFILES.items():
        result = core.predict(profile)
        top = sorted(result.probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
        demos[title] = {"label": result.label, "top3": dict(top)}
        print(f"   {title}: {result.label}  ->  "
              + ", ".join(f"{name} {prob:.0%}" for name, prob in top))

    report = classification_report(true_kept, pred_kept, labels=names, output_dict=True)
    OUTPUT_JSON.write_text(json.dumps({
        "split": {"dataset": "UCI 544", "deduplicated": True, "test_size": TEST_SIZE,
                  "stratified": True, "random_state": SEED},
        "rows_evaluated": len(kept),
        "rows_rejected": len(rejected),
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "notebook_accuracy": reference["accuracy"],
        "notebook_f1_macro": reference["f1_macro"],
        "ms_per_person": round(elapsed_ms, 2),
        "per_class": {name: {k: round(v, 3) for k, v in report[name].items()} for name in names},
        "confusion_matrix": {"labels": names, "matrix": matrix.tolist()},
        "demo_profiles": demos,
    }, indent=2), encoding="utf-8")
    save_confusion_png(matrix, names)
    print(f"\nSaved {OUTPUT_JSON.relative_to(REPO_ROOT)} and {OUTPUT_PNG.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
