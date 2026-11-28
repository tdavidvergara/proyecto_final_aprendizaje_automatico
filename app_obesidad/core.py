"""Model loading, input encoding and prediction for the obesity-level desktop app.

The encoding reproduces cell 32 of `Notebook_Matriz.ipynb`, which turned the raw UCI survey
answers into the 17 preventive columns the winning pipeline was trained on:

- Gender: Female -> 0, Male -> 1.
- family_history_with_overweight, FAVC, SMOKE, SCC: no -> 0, yes -> 1.
- FCVC, NCP, CH2O, FAF, TUE: rounded to the nearest integer of the survey scale.
- CAEC, CALC: no -> 0, Sometimes -> 1, Frequently -> 2, Always -> 3
  (CALC "Always" is merged into "Frequently" first, as in the notebook).
- MTRANS: Bike and Motorbike become Two_Wheels, then one-hot encoded into
  MTRANS_Automobile, MTRANS_Public_Transportation, MTRANS_Two_Wheels, MTRANS_Walking.

Values outside the training range stored in `metricas_ganador.json` are rejected with a
ValueError instead of being clamped, so the app never silently changes what the user typed.
"""
from __future__ import annotations

import contextlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, NamedTuple

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "modelos"
MODEL_PATH = MODELS_DIR / "modelo_ganador.pkl"
METADATA_PATH = MODELS_DIR / "metricas_ganador.json"

# Older pickles reference `src.p2_habitos.agregar_habitos`; keep the repo root importable.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import p2_habitos  # noqa: E402  (needs the repo root on sys.path)

# The current winner was pickled from `Notebook_Matriz.ipynb`, so its P2 FunctionTransformer
# references `__main__.agregar_habitos` and `__main__.nombres_con_habitos`
# (see docs/GUIA_DESPLIEGUE_INTERFAZ.md, section 4).
P2_MAIN_CALLABLES: Dict[str, Any] = {
    "agregar_habitos": p2_habitos.agregar_habitos,
    "nombres_con_habitos": p2_habitos._nombres_con_habitos,
}

BINARY_MAPS: Dict[str, Dict[str, int]] = {
    "Gender": {"Female": 0, "Male": 1},
    "family_history_with_overweight": {"no": 0, "yes": 1},
    "FAVC": {"no": 0, "yes": 1},
    "SMOKE": {"no": 0, "yes": 1},
    "SCC": {"no": 0, "yes": 1},
}
FREQUENCY_ORDER: Dict[str, int] = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
CALC_MERGES: Dict[str, str] = {"Always": "Frequently"}
ORDINAL_SCALES = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
TRANSPORT_MERGES: Dict[str, str] = {"Bike": "Two_Wheels", "Motorbike": "Two_Wheels"}
TRANSPORT_OPTIONS = ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
TRANSPORT_COLUMNS = {
    "Automobile": "MTRANS_Automobile",
    "Public_Transportation": "MTRANS_Public_Transportation",
    "Two_Wheels": "MTRANS_Two_Wheels",
    "Walking": "MTRANS_Walking",
}
INPUT_FIELDS = [
    "Gender", "Age", "family_history_with_overweight", "FAVC", "FCVC", "NCP", "CAEC",
    "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS",
]

_SERIALIZED_MODEL_KEY = b"L\x00\x00\x00\x00\x00\x00\x00\x05Model{"


class Prediction(NamedTuple):
    """Predicted class name and the probability of each of the 7 classes."""

    label: str
    probabilities: Dict[str, float]


@contextlib.contextmanager
def _portable_xgboost_unpickling() -> Iterator[None]:
    """Let XGBoost boosters pickled on Linux load on Windows.

    An XGBoost pickle stores the booster as a UBJSON document with a "Config" section and a
    "Model" section. The Config holds the training RNG state, serialized by the C++ standard
    library of the platform that pickled it; libstdc++ (Linux/Colab) and MSVC (Windows) use
    different formats, so `XGBoosterUnserializeFromBuffer` rejects it ("input stream
    corrupted"). Prediction only needs the "Model" section, which is the portable format used
    by `Booster.save_model`, so on failure the booster is rebuilt from that section alone.
    """
    import xgboost
    from xgboost.core import Booster

    original_setstate = Booster.__setstate__

    def tolerant_setstate(self: Booster, state: Dict[str, Any]) -> None:
        try:
            original_setstate(self, state)
            return
        except xgboost.core.XGBoostError:
            raw = bytes(state["handle"])
            start = raw.find(_SERIALIZED_MODEL_KEY)
            if start < 0:
                raise
        model_section = raw[start + len(_SERIALIZED_MODEL_KEY) - 1:-1]
        rebuilt = Booster()
        rebuilt.load_model(bytearray(model_section))
        self.__dict__.update({k: v for k, v in state.items() if k != "handle"})
        self.handle = rebuilt.handle
        rebuilt.handle = None  # ownership moves to self; avoid a double free

    Booster.__setstate__ = tolerant_setstate
    try:
        yield
    finally:
        Booster.__setstate__ = original_setstate


def register_p2_on_main() -> None:
    """Expose the P2 callables on `__main__` so notebook-pickled pipelines can be unpickled.

    Attributes that `__main__` already defines are left untouched.
    """
    main_module = sys.modules["__main__"]
    for name, function in P2_MAIN_CALLABLES.items():
        if not hasattr(main_module, name):
            setattr(main_module, name, function)


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH):
    """Load the winning sklearn pipeline (cached after the first call)."""
    register_p2_on_main()
    with _portable_xgboost_unpickling():
        return joblib.load(path)


@lru_cache(maxsize=1)
def load_metadata(path: Path = METADATA_PATH) -> Dict[str, Any]:
    """Load `metricas_ganador.json` (metrics, columns, classes and training ranges)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def class_names() -> List[str]:
    """Class names ordered by their encoded label (0 = Insufficient_Weight ... 6 = Obesity_Type_III)."""
    classes = load_metadata()["clases"]
    return [classes[key] for key in sorted(classes, key=int)]


def _check_range(column: str, value: float) -> None:
    low, high = load_metadata()["rangos"][column]
    if not low <= value <= high:
        raise ValueError(f"{column}={value} is outside the training range [{low}, {high}]")


def _lookup(column: str, value: Any, mapping: Mapping[str, int]) -> int:
    if value not in mapping:
        raise ValueError(f"{column}={value!r} is not one of {sorted(mapping)}")
    return mapping[value]


def encode(profile: Mapping[str, Any]) -> pd.DataFrame:
    """Turn raw survey answers (UCI dataset vocabulary) into the model's 17-column row.

    Raises ValueError for missing fields, unknown categories or out-of-range values.
    """
    missing = [field for field in INPUT_FIELDS if field not in profile]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    row: Dict[str, float] = {}
    for column, mapping in BINARY_MAPS.items():
        row[column] = _lookup(column, profile[column], mapping)

    row["Age"] = float(profile["Age"])
    for column in ORDINAL_SCALES:
        value = float(profile[column])
        _check_range(column, value)  # before rounding, so e.g. 3.4 is not accepted as 3
        row[column] = int(round(value))

    row["CAEC"] = _lookup("CAEC", profile["CAEC"], FREQUENCY_ORDER)
    calc = CALC_MERGES.get(profile["CALC"], profile["CALC"])
    row["CALC"] = _lookup("CALC", calc, FREQUENCY_ORDER)

    if profile["MTRANS"] not in TRANSPORT_OPTIONS:
        raise ValueError(f"MTRANS={profile['MTRANS']!r} is not one of {TRANSPORT_OPTIONS}")
    transport = TRANSPORT_MERGES.get(profile["MTRANS"], profile["MTRANS"])
    for option, column in TRANSPORT_COLUMNS.items():
        row[column] = int(option == transport)

    columns = load_metadata()["columnas"]
    for column in columns:
        _check_range(column, row[column])

    frame = pd.DataFrame([row], columns=columns)
    int_columns = [c for c in columns if c != "Age"]
    return frame.astype({c: "int64" for c in int_columns} | {"Age": "float64"})


def predict(profile: Mapping[str, Any]) -> Prediction:
    """Classify one profile and return the predicted class plus all 7 probabilities."""
    model = load_model()
    probabilities = model.predict_proba(encode(profile))[0]
    names = class_names()
    labels = [int(label) for label in model.classes_]
    by_class = {names[label]: float(probabilities[i]) for i, label in enumerate(labels)}
    ordered = {name: by_class[name] for name in names}
    return Prediction(label=max(ordered, key=ordered.get), probabilities=ordered)
