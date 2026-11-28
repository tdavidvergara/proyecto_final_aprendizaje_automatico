"""Metodología de preprocesamiento P2: Ingeniería de Hábitos + Yeo-Johnson + StandardScaler.

P2 está pensada para el benchmark preventivo (17 variables, sin Weight, Height ni IMC):
1. Agrega 4 variables de dominio construidas solo con hábitos (fila a fila, sin parámetros
   aprendidos, por lo que no hay fuga entre entrenamiento y prueba).
2. Corrige la asimetría de las variables no binarias con Yeo-Johnson.
3. Escala todas las variables (21 en total) a media 0 y desviación 1.

Todo se encapsula en un `Pipeline` de sklearn para que se ajuste solo con entrenamiento.
"""
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

HABIT_FEATURES = ["Balance_Act_Sed", "Hidratacion_Activa", "Riesgo_Gen_Cal", "Indice_Protector"]

# Variables antropométricas: la clase se calcula con ellas, así que su presencia sería fuga
FORBIDDEN_COLUMNS = ["Weight", "Height", "IMC"]


def agregar_habitos(X: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de X con las 4 variables de hábitos agregadas.

    - Balance_Act_Sed    = FAF / (TUE + 1): actividad física frente a tiempo frente a pantallas.
    - Hidratacion_Activa = CH2O * (FAF + 1): consumo de agua potenciado por la actividad física.
      (No se divide por Weight: eso filtra la clase).
    - Riesgo_Gen_Cal     = family_history_with_overweight * FAVC: antecedente familiar y
      consumo frecuente de comida calórica a la vez.
    - Indice_Protector   = FCVC + CH2O + FAF + SCC - FAVC: suma de hábitos protectores menos
      el consumo calórico. Los signos son supuestos porque la fórmula original llegó truncada;
      pendiente de confirmar con Sebastián.

    Lanza ValueError si X contiene Weight, Height o IMC: P2 es solo para el benchmark
    preventivo y así se evita la fuga de información.
    """
    presentes = [c for c in FORBIDDEN_COLUMNS if c in X.columns]
    if presentes:
        raise ValueError(
            f"P2 no admite variables antropométricas (fuga de la clase): {presentes}")

    X = X.copy()
    X["Balance_Act_Sed"] = X["FAF"] / (X["TUE"] + 1)
    X["Hidratacion_Activa"] = X["CH2O"] * (X["FAF"] + 1)
    X["Riesgo_Gen_Cal"] = X["family_history_with_overweight"] * X["FAVC"]
    X["Indice_Protector"] = X["FCVC"] + X["CH2O"] + X["FAF"] + X["SCC"] - X["FAVC"]
    return X


def _nombres_con_habitos(transformer, input_features) -> List[str]:
    """Nombres de salida del paso de hábitos: las columnas de entrada más las 4 nuevas."""
    return list(input_features) + HABIT_FEATURES


def columnas_continuas(X: pd.DataFrame) -> List[str]:
    """Columnas con más de 2 valores distintos (no binarias ni one-hot)."""
    return [c for c in X.columns if X[c].nunique() > 2]


def construir_p2(X_train_ejemplo: pd.DataFrame) -> Pipeline:
    """Construye (sin ajustar) el pipeline P2.

    Escala con StandardScaler (21 variables): las continuas pasan por Yeo-Johnson con
    `standardize=True`, que ya las deja con media 0 y desviación 1; el resto (binarias y
    one-hot) pasa por StandardScaler. Las columnas continuas se detectan sobre
    `X_train_ejemplo` (con los hábitos agregados), por lo que debe ser el conjunto de
    entrenamiento. El pipeline se ajusta después con `fit(X_train)`.
    """
    continuas = columnas_continuas(agregar_habitos(X_train_ejemplo))
    transformacion = ColumnTransformer(
        [("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=True), continuas)],
        remainder=StandardScaler(),
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("habitos", FunctionTransformer(agregar_habitos, feature_names_out=_nombres_con_habitos)),
        ("transformacion", transformacion),
    ])
