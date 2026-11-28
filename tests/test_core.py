"""Tests for the desktop app core: model loading, input encoding and prediction.

The encoding under test mirrors cell 32 of `Notebook_Matriz.ipynb` (same as `Desarrollo.ipynb`),
which produced the 17 preventive columns the winning pipeline was trained on.
"""
import math

import pandas as pd
import pytest

from app_obesidad import core

SAMPLE_PROFILE = {
    "Gender": "Male",
    "Age": 25,
    "family_history_with_overweight": "yes",
    "FAVC": "yes",
    "FCVC": 2,
    "NCP": 3,
    "CAEC": "Sometimes",
    "SMOKE": "no",
    "CH2O": 2,
    "SCC": "no",
    "FAF": 1,
    "TUE": 1,
    "CALC": "Sometimes",
    "MTRANS": "Public_Transportation",
}


def test_model_and_metadata_load():
    model = core.load_model()
    metadata = core.load_metadata()
    assert hasattr(model, "predict_proba")
    assert "clf" in model.named_steps
    assert len(metadata["columnas"]) == 17
    assert len(metadata["clases"]) == 7


def test_p2_callables_are_registered_on_main_without_clobbering(monkeypatch):
    import sys

    main_module = sys.modules["__main__"]
    sentinel = object()
    monkeypatch.delattr(main_module, "agregar_habitos", raising=False)
    monkeypatch.setattr(main_module, "nombres_con_habitos", sentinel, raising=False)
    core.register_p2_on_main()
    assert main_module.agregar_habitos is core.P2_MAIN_CALLABLES["agregar_habitos"]
    assert main_module.nombres_con_habitos is sentinel


def test_class_names_follow_metadata_order():
    assert core.class_names() == [
        "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II",
        "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
    ]


def test_encode_produces_the_17_model_columns_in_order():
    row = core.encode(SAMPLE_PROFILE)
    assert isinstance(row, pd.DataFrame)
    assert row.shape == (1, 17)
    assert list(row.columns) == core.load_metadata()["columnas"]


def test_encode_matches_notebook_mapping():
    row = core.encode(SAMPLE_PROFILE).iloc[0]
    assert row["Gender"] == 1  # Female -> 0, Male -> 1
    assert row["Age"] == 25.0
    assert row["family_history_with_overweight"] == 1
    assert row["FAVC"] == 1
    assert row["SMOKE"] == 0
    assert row["SCC"] == 0
    assert row["CAEC"] == 1  # no=0, Sometimes=1, Frequently=2, Always=3
    assert row["CALC"] == 1
    assert row[["MTRANS_Automobile", "MTRANS_Public_Transportation",
                "MTRANS_Two_Wheels", "MTRANS_Walking"]].tolist() == [0, 1, 0, 0]


@pytest.mark.parametrize("transport", ["Bike", "Motorbike"])
def test_bike_and_motorbike_map_to_two_wheels(transport):
    row = core.encode({**SAMPLE_PROFILE, "MTRANS": transport}).iloc[0]
    assert row["MTRANS_Two_Wheels"] == 1
    assert row[["MTRANS_Automobile", "MTRANS_Public_Transportation", "MTRANS_Walking"]].sum() == 0


def test_calc_always_is_merged_into_frequently():
    row = core.encode({**SAMPLE_PROFILE, "CALC": "Always"}).iloc[0]
    assert row["CALC"] == 2


def test_survey_scales_are_rounded_to_integers():
    row = core.encode({**SAMPLE_PROFILE, "FCVC": 2.6, "FAF": 0.4}).iloc[0]
    assert row["FCVC"] == 3
    assert row["FAF"] == 0


@pytest.mark.parametrize("field, value", [
    ("Age", 13),
    ("Age", 70),
    ("NCP", 5),
    ("TUE", 3),
    ("FCVC", 0),
    ("FCVC", 3.4),  # would round to 3 (in range) if rounding ran before the range check
    ("FAF", -0.4),  # would round to 0 (in range)
])
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(ValueError):
        core.encode({**SAMPLE_PROFILE, field: value})


@pytest.mark.parametrize("field, value", [
    ("Gender", "Other"),
    ("FAVC", "maybe"),
    ("CAEC", "Never"),
    ("MTRANS", "Train"),
])
def test_unknown_categories_are_rejected(field, value):
    with pytest.raises(ValueError):
        core.encode({**SAMPLE_PROFILE, field: value})


def test_missing_field_is_rejected():
    profile = dict(SAMPLE_PROFILE)
    del profile["SMOKE"]
    with pytest.raises(ValueError):
        core.encode(profile)


def test_predict_returns_label_and_seven_probabilities():
    result = core.predict(SAMPLE_PROFILE)
    assert result.label in core.class_names()
    assert list(result.probabilities) == core.class_names()
    assert len(result.probabilities) == 7
    assert all(0.0 <= p <= 1.0 for p in result.probabilities.values())
    assert math.isclose(sum(result.probabilities.values()), 1.0, abs_tol=1e-4)
    assert result.label == max(result.probabilities, key=result.probabilities.get)


def test_predict_matches_pipeline_predict():
    model = core.load_model()
    expected = core.class_names()[int(model.predict(core.encode(SAMPLE_PROFILE))[0])]
    assert core.predict(SAMPLE_PROFILE).label == expected


# Regression snapshot, NOT independent ground truth: neither Notebook_Matriz.ipynb nor
# docs/GUIA_DESPLIEGUE_INTERFAZ.md publishes a per-profile prediction to compare against.
# These values were recorded from the retrained winner (metricas_ganador.json fecha
# 2026-11-28 07:17) and catch accidental changes to encoding, loading or the pickle itself.
OBESE_PROFILE = {
    **SAMPLE_PROFILE, "Gender": "Female", "Age": 26, "FCVC": 3, "CH2O": 1, "FAF": 0, "TUE": 0,
}
SNAPSHOTS = [
    (SAMPLE_PROFILE, "Normal_Weight", [0.0003, 0.7463, 0.0078, 0.1036, 0.0022, 0.1394, 0.0004]),
    (OBESE_PROFILE, "Obesity_Type_III", [0.0001, 0.0008, 0.0003, 0.0002, 0.0018, 0.0007, 0.9961]),
]


@pytest.mark.parametrize("profile, label, probabilities", SNAPSHOTS)
def test_prediction_regression_snapshot(profile, label, probabilities):
    result = core.predict(profile)
    assert result.label == label
    assert list(result.probabilities.values()) == pytest.approx(probabilities, abs=1e-3)


def test_booster_rebuilt_from_model_section_predicts_identically(monkeypatch):
    """Force the Windows fallback on the real winner's booster and compare predictions."""
    import pickle

    import xgboost
    from xgboost.core import Booster

    pipeline = core.load_model()
    booster = pipeline.named_steps["clf"].get_booster()
    features = xgboost.DMatrix(pipeline[:-1].transform(core.encode(SAMPLE_PROFILE)))
    expected = booster.predict(features)
    pickled = pickle.dumps(booster)
    assert core._SERIALIZED_MODEL_KEY in pickled

    def corrupted_stream(self, state):
        raise xgboost.core.XGBoostError("input stream corrupted")

    monkeypatch.setattr(Booster, "__setstate__", corrupted_stream)
    with core._portable_xgboost_unpickling():
        rebuilt = pickle.loads(pickled)
    assert Booster.__setstate__ is corrupted_stream  # the context restores what it replaced
    monkeypatch.undo()

    assert rebuilt.predict(features).tolist() == expected.tolist()
