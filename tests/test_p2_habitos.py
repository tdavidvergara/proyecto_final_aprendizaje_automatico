"""Pruebas de la metodología de preprocesamiento P2 (ingeniería de hábitos + Yeo-Johnson + StandardScaler).

Todas las pruebas son offline: usan un DataFrame sintético con las 17 columnas preventivas
codificadas igual que en `Desarrollo.ipynb`.
"""
import numpy as np
import pandas as pd
import pytest

from src.p2_habitos import (
    HABIT_FEATURES,
    agregar_habitos,
    columnas_continuas,
    construir_p2,
)

PREVENTIVE_COLUMNS = [
    "Gender", "Age", "family_history_with_overweight", "FAVC", "FCVC", "NCP", "CAEC",
    "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC",
    "MTRANS_Automobile", "MTRANS_Public_Transportation", "MTRANS_Two_Wheels", "MTRANS_Walking",
]


def make_preventive(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Genera datos sintéticos con rangos realistas para las 17 columnas preventivas."""
    rng = np.random.default_rng(seed)
    transport = rng.choice(4, size=n, p=[0.25, 0.6, 0.05, 0.1])
    data = {
        "Gender": rng.integers(0, 2, n),
        # Edad con asimetría a la derecha, como en el dataset real
        "Age": 14 + rng.lognormal(mean=2.2, sigma=0.5, size=n),
        "family_history_with_overweight": rng.integers(0, 2, n),
        "FAVC": rng.integers(0, 2, n),
        "FCVC": rng.integers(1, 4, n),
        "NCP": rng.integers(1, 5, n),
        "CAEC": rng.integers(0, 4, n),
        "SMOKE": rng.integers(0, 2, n),
        "CH2O": rng.integers(1, 4, n),
        "SCC": rng.integers(0, 2, n),
        "FAF": rng.choice(4, size=n, p=[0.4, 0.3, 0.2, 0.1]),
        "TUE": rng.integers(0, 3, n),
        "CALC": rng.integers(0, 3, n),
        "MTRANS_Automobile": (transport == 0).astype(int),
        "MTRANS_Public_Transportation": (transport == 1).astype(int),
        "MTRANS_Two_Wheels": (transport == 2).astype(int),
        "MTRANS_Walking": (transport == 3).astype(int),
    }
    return pd.DataFrame(data, columns=PREVENTIVE_COLUMNS)


@pytest.fixture
def train_test():
    X = make_preventive(500, seed=1)
    return X.iloc[:400].copy(), X.iloc[400:].copy()


def test_habit_features_exact_values_on_hand_built_row():
    row = make_preventive(1).iloc[[0]].copy()
    row[["FAF", "TUE", "CH2O", "family_history_with_overweight", "FAVC", "FCVC", "SCC"]] = [
        2, 1, 3, 1, 1, 2, 1,
    ]
    out = agregar_habitos(row).iloc[0]
    assert out["Balance_Act_Sed"] == pytest.approx(2 / (1 + 1))
    assert out["Hidratacion_Activa"] == pytest.approx(3 * (2 + 1))
    assert out["Riesgo_Gen_Cal"] == pytest.approx(1 * 1)
    assert out["Indice_Protector"] == pytest.approx(2 + 3 + 2 + 1 - 1)


def test_agregar_habitos_returns_copy_with_four_new_columns():
    X = make_preventive(10)
    out = agregar_habitos(X)
    assert list(out.columns) == PREVENTIVE_COLUMNS + HABIT_FEATURES
    assert list(X.columns) == PREVENTIVE_COLUMNS  # la entrada no se modifica


@pytest.mark.parametrize("forbidden", ["Weight", "Height", "IMC"])
def test_agregar_habitos_rejects_anthropometric_columns(forbidden):
    X = make_preventive(10)
    X[forbidden] = 1.0
    with pytest.raises(ValueError):
        agregar_habitos(X)


def test_columnas_continuas_excludes_binary_and_one_hot():
    continuous = columnas_continuas(agregar_habitos(make_preventive(300)))
    assert "Age" in continuous
    assert {"Balance_Act_Sed", "Hidratacion_Activa", "Indice_Protector"} <= set(continuous)
    for binary in ["Gender", "FAVC", "SMOKE", "SCC", "family_history_with_overweight",
                   "Riesgo_Gen_Cal", "MTRANS_Automobile", "MTRANS_Walking"]:
        assert binary not in continuous


def test_pipeline_outputs_21_named_standardized_columns(train_test):
    X_train, _ = train_test
    p2 = construir_p2(X_train).set_output(transform="pandas")
    Z = p2.fit_transform(X_train)
    names = list(p2.get_feature_names_out())
    assert Z.shape == (len(X_train), 21)
    assert len(names) == 21
    assert set(names) == set(PREVENTIVE_COLUMNS + HABIT_FEATURES)
    assert list(Z.columns) == names
    np.testing.assert_allclose(Z.mean().to_numpy(), 0, atol=1e-8)
    np.testing.assert_allclose(Z.std(ddof=0).to_numpy(), 1, atol=1e-6)


def _lambdas(pipeline):
    return pipeline.named_steps["transformacion"].named_transformers_["yeo_johnson"].lambdas_.copy()


def test_transforming_test_does_not_refit(train_test):
    X_train, X_test = train_test
    p2 = construir_p2(X_train).fit(X_train)
    before = _lambdas(p2)
    p2.transform(X_test)
    np.testing.assert_array_equal(before, _lambdas(p2))


def test_fit_on_train_only_differs_from_fit_on_all_data(train_test):
    X_train, X_test = train_test
    X_all = pd.concat([X_train, X_test])
    lambdas_train = _lambdas(construir_p2(X_train).fit(X_train))
    lambdas_all = _lambdas(construir_p2(X_train).fit(X_all))
    continuous = columnas_continuas(agregar_habitos(X_train))
    age = continuous.index("Age")
    assert lambdas_train[age] != pytest.approx(lambdas_all[age], abs=1e-6)
