# Guía para la Interfaz y Despliegue (`modelo_ganador.pkl`)

Documento de entrega para el equipo encargado de la **interfaz gráfica / despliegue** del proyecto final (*Estimación de Niveles de Obesidad*, UCI ID `544`).

---

## 1. Qué se hizo en el `Notebook_Matriz.ipynb`

El cuaderno principal [Notebook_Matriz.ipynb](../Notebook_Matriz.ipynb) ejecuta de forma reproducible (semilla fija `SEMILLA = 20`, 0 errores) todo el experimento $3 \times 3$:

1. **Datos y limpieza:** descarga oficial desde UCI (`id=544`), codificación consistente con `Desarrollo.ipynb`, eliminación de 24 duplicados (`2,111 → 2,087` registros) y división **80/20 estratificada** (`1,669` entrenamiento / `418` prueba).
2. **Benchmark principal (Preventivo):** utiliza **únicamente las 17 variables de hábitos y demografía** (sin `Weight`, `Height` ni `IMC`), porque la etiqueta `NObeyesdad` se construye matemáticamente con el IMC ($\text{Peso}/\text{Altura}^2$) y meter esas columnas produciría fuga de datos (*data leakage*).
3. **Tres metodologías de preprocesamiento (ajustadas dentro de cada pliegue):**
   - **P1 (`P1_RobustScaler`):** 17 variables escaladas con mediana e IQR.
   - **P2 (`P2_Habitos_YeoJohnson`):** agrega 4 variables derivadas de hábitos sin peso/estatura (`Balance_Act_Sed`, `Hidratacion_Activa`, `Riesgo_Gen_Cal`, `Indice_Protector`), aplica transformación Yeo-Johnson a las continuas y `StandardScaler` al resto (21 variables).
   - **P3 (`P3_Robust_PCA95`):** `RobustScaler` + `PCA` reteniendo el 95 % de varianza (17 → 11 componentes, 95.76 % varianza).
4. **Tres modelos supervisados (`StratifiedKFold(5)`, `RandomizedSearchCV`, `f1_macro`):**
   - **M1:** `RandomForestClassifier` (*Bagging*)
   - **M2:** `XGBClassifier` (*Gradient Boosting*)
   - **M3:** `MLPClassifier` (*Red Neuronal Multicapa*)
5. **Controles científicos y evaluación final (Paso 9):** control `sin Gender` (sesgo SMOTE), control `Operativo` (fuga de IMC), F1 por cada una de las 7 clases, `Precision`/`Recall`/`ROC-AUC OvR`, curvas de aprendizaje (`learning_curve`) y análisis de complejidad teórica (Big-O) y empírica (tiempo de fit, latencia y tamaño `.pkl`).

---

## 2. Cuál fue el modelo ganador y por qué

### Ganador oficial (seleccionado por validación cruzada sin mirar el test)
- **Algoritmo:** **`M2 · XGBoost (`XGBClassifier`)`**
- **Preprocesamiento:** **`P2 · Ingeniería de Hábitos + Yeo-Johnson + StandardScaler`**
- **Archivo listo para producción:** `modelos/modelo_ganador.pkl` (incluye el *pipeline* completo: preprocesamiento `P2` + clasificador `XGBoost`)
- **Metadatos y rangos para la UI:** `modelos/metricas_ganador.json`

### Comparativa final de los 3 finalistas (`modelos/tabla_decision_finalistas.csv`)

| Modelo | Mejor Prep (CV) | CV F1-macro | Test Acc | Test F1 | Precision | Recall | ROC-AUC (OvR) | F1 sin Gender | Fit (s) | ms / muestra | Tamaño `.pkl` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M2 · XGBoost** 🏆 | **`P2_Habitos_YeoJohnson`** | **0.7943 ± 0.0114** | **79.67 %** | **0.7939** | **0.7948** | **0.7938** | **0.9638** | **0.7867** | **0.69 s** | 0.057 ms | 3,444.5 KB |
| M1 · Random Forest | `P1_RobustScaler` | 0.7749 ± 0.0143 | 78.95 % | 0.7819 | 0.7833 | 0.7867 | 0.9560 | 0.7347 | 0.93 s | 0.263 ms | 9,955.7 KB |
| M3 · MLP | `P2_Habitos_YeoJohnson` | 0.7491 ± 0.0027 | 75.60 % | 0.7518 | 0.7533 | 0.7536 | 0.9416 | 0.7153 | 3.53 s | 0.027 ms | 937.3 KB |

### Por qué se eligió `M2 · XGBoost + P2`
1. **Mejor desempeño en todas las métricas:** líder en `CV F1-macro` (`0.7943`), `Test Accuracy` (`79.67 %`), `Test F1-macro` (`0.7939`), `Precision`, `Recall` y `ROC-AUC OvR` (`0.9638`).
2. **Mayor robustez ante el sesgo de género (SMOTE):** aun quitando `Gender`, mantiene `0.7867` de F1-macro (por encima de Random Forest y MLP *con* `Gender`).
3. **Excelente balance de complejidad:** entrena en solo `0.69 s`, pesa **3× menos** que Random Forest (`3.4 MB` vs. `10 MB`) y predice **4.6× más rápido** (`0.057 ms` por paciente).

> **Alternativa sin dependencia de funciones custom (si quieren simplificar aún más la app):** en la carpeta `modelos/` también está `m2_xgb_p1_robustscaler_preventivo.pkl` (`M2 · XGBoost + P1`), que empató técnicamente en CV (`0.7942 ± 0.0154`), obtuvo `Test F1 = 0.8203` (`Test Acc = 82.30 %`) y usa únicamente `RobustScaler` estándar de `scikit-learn` (no requiere registrar funciones de P2 al cargar con `joblib`). Sin embargo, el ganador oficial reportado por CV es **`modelo_ganador.pkl` (`M2 + P2`)**.

---

## 3. Contrato exacto de entrada para la interfaz (17 columnas)

El `Pipeline` guardado en `modelos/modelo_ganador.pkl` espera un **`pandas.DataFrame` con estas 17 columnas exactas y en este mismo orden**:

| # | Columna | Qué pregunta al usuario | Tipo / Codificación exacta | Rango válido |
| :-: | :--- | :--- | :--- | :---: |
| 1 | `Gender` | Género | `0` = Mujer · `1` = Hombre | `0` o `1` |
| 2 | `Age` | Edad (años) | Numérico (`float` o `int`) | `14` – `61` |
| 3 | `family_history_with_overweight` | ¿Antecedentes familiares de sobrepeso? | `0` = No · `1` = Sí | `0` o `1` |
| 4 | `FAVC` | ¿Consume comida alta en calorías frecuentemente? | `0` = No · `1` = Sí | `0` o `1` |
| 5 | `FCVC` | Frecuencia de consumo de verduras | `1` = Nunca · `2` = A veces · `3` = Siempre | `1`, `2` o `3` |
| 6 | `NCP` | Comidas principales al día | `1` = 1 · `2` = 2 · `3` = 3 · `4` = 4 o más | `1` – `4` |
| 7 | `CAEC` | ¿Come alimentos entre comidas? | `0` = No · `1` = A veces · `2` = Frecuentemente · `3` = Siempre | `0` – `3` |
| 8 | `SMOKE` | ¿Fuma? | `0` = No · `1` = Sí | `0` o `1` |
| 9 | `CH2O` | Consumo diario de agua | `1` = < 1 L · `2` = 1–2 L · `3` = > 2 L | `1`, `2` o `3` |
| 10 | `SCC` | ¿Monitorea las calorías que consume? | `0` = No · `1` = Sí | `0` o `1` |
| 11 | `FAF` | Frecuencia semanal de actividad física | `0` = Ninguna · `1` = 1–2 días · `2` = 3–4 días · `3` = 5+ días | `0` – `3` |
| 12 | `TUE` | Tiempo diario usando dispositivos tecnológicos | `0` = 0–2 h · `1` = 3–5 h · `2` = > 5 h | `0`, `1` o `2` |
| 13 | `CALC` | Frecuencia de consumo de alcohol | `0` = No bebe · `1` = A veces · `2` = Frecuentemente / Siempre | `0`, `1` o `2` |
| 14 | `MTRANS_Automobile` | Transporte principal = Automóvil | `1` si eligió esta opción, `0` si no | `0` o `1` |
| 15 | `MTRANS_Public_Transportation` | Transporte principal = Transporte público | `1` si eligió esta opción, `0` si no | `0` o `1` |
| 16 | `MTRANS_Two_Wheels` | Transporte principal = Bicicleta o Moto | `1` si eligió esta opción, `0` si no | `0` o `1` |
| 17 | `MTRANS_Walking` | Transporte principal = Caminando | `1` si eligió esta opción, `0` si no | `0` o `1` |

> **Importante sobre `MTRANS`:** en la interfaz muestren **un solo menú desplegable** ("Medio de transporte principal") con las 4 opciones, y en el código conviertan la selección en las 4 columnas `0/1` (exactamente una en `1` y las otras tres en `0`).

---

## 4. Cómo cargar y predecir sin errores (código listo para usar)

Como `modelo_ganador.pkl` usa el preprocesamiento **P2** definido en `Notebook_Matriz.ipynb`, al cargarlo desde un script externo (`app.py` / `core.py`) asegúrense de exponer las funciones `agregar_habitos` y `nombres_con_habitos` antes de llamar a `joblib.load()` (o usen el bloque exacto de abajo, que funciona siempre):

```python
import json
import sys
import joblib
import pandas as pd

# 1. Registrar las funciones de P2 para que joblib cargue modelo_ganador.pkl sin errores
HABIT_FEATURES = ["Balance_Act_Sed", "Hidratacion_Activa", "Riesgo_Gen_Cal", "Indice_Protector"]
FORBIDDEN_COLUMNS = ["Weight", "Height", "IMC"]

def agregar_habitos(X):
    presentes = [c for c in FORBIDDEN_COLUMNS if c in X.columns]
    if presentes:
        raise ValueError(f"P2 no admite variables antropométricas: {presentes}")
    X = X.copy()
    X["Balance_Act_Sed"] = X["FAF"] / (X["TUE"] + 1)
    X["Hidratacion_Activa"] = X["CH2O"] * (X["FAF"] + 1)
    X["Riesgo_Gen_Cal"] = X["family_history_with_overweight"] * X["FAVC"]
    X["Indice_Protector"] = X["FCVC"] + X["CH2O"] + X["FAF"] + X["SCC"] - X["FAVC"]
    return X

def nombres_con_habitos(transformer, input_features):
    return list(input_features) + HABIT_FEATURES

# Asegurar que estén disponibles en __main__ incluso si este código vive en core.py
main_mod = sys.modules["__main__"]
setattr(main_mod, "agregar_habitos", agregar_habitos)
setattr(main_mod, "nombres_con_habitos", nombres_con_habitos)

# 2. Cargar modelo ganador y metadatos (mismo patrón de Semana 7)
modelo = joblib.load("modelos/modelo_ganador.pkl")
with open("modelos/metricas_ganador.json", encoding="utf-8") as f:
    metricas = json.load(f)

ETIQUETAS_ES = {
    0: "Peso insuficiente",
    1: "Peso normal",
    2: "Sobrepeso nivel I",
    3: "Sobrepeso nivel II",
    4: "Obesidad tipo I",
    5: "Obesidad tipo II",
    6: "Obesidad tipo III",
}

def predecir_paciente(respuestas_dict):
    """Recibe un diccionario con las 17 columnas y devuelve clase + probabilidades."""
    df_entrada = pd.DataFrame([respuestas_dict])[metricas["columnas"]]
    clase_id = int(modelo.predict(df_entrada)[0])
    probs = modelo.predict_proba(df_entrada)[0]
    return {
        "clase_id": clase_id,
        "etiqueta": ETIQUETAS_ES[clase_id],
        "confianza": float(probs[clase_id]),
        "probabilidades": {ETIQUETAS_ES[i]: float(p) for i, p in enumerate(probs)},
    }
```

---

## 5. Recomendaciones clave para una excelente implementación

1. **NO pasar `Weight`, `Height` ni `IMC` al modelo:** si las pasan, `P2` lanzará un `ValueError` adrede para evitar fuga de datos. La app es un **tamizaje preventivo basado en hábitos**.
2. **Mostrar las probabilidades de las 7 clases (`predict_proba`):** además del nivel predicho, muestren una barra de confianza o el Top-3 de probabilidades; en casos frontera (p. ej. *Sobrepeso I* vs. *Sobrepeso II*) se entiende mucho mejor la predicción en la demo.
3. **Leer `modelos/metricas_ganador.json` en la interfaz:** muestren en el pie de página o panel lateral el algoritmo activo (`M2_XGBoost + P2_Habitos_YeoJohnson`), su exactitud (`79.67 %`) y F1-macro (`0.7939`), tal como en el proyecto guía de Semana 7 del profesor.
4. **Incluir un aviso de uso responsable (limitación del dataset):** agreguen una nota visible indicando que es una herramienta académica de tamizaje (no diagnóstico médico) y que el dataset original contiene datos sintéticos (SMOTE) con sesgo de género en Obesidad II y III.
5. **Dependencias mínimas para correr la app:** `scikit-learn`, `xgboost`, `pandas`, `numpy`, `joblib` (más la librería de interfaz que escojan, p. ej. `customtkinter` o `streamlit`).
6. **Gráficas listas para diapositivas / paper en `modelos/`:**
   - `modelos/matriz_3x3_comparativa.png`
   - `modelos/comparacion_f1_por_clase.png`
   - `modelos/curvas_aprendizaje_finalistas.png`
   - `modelos/escalabilidad_tiempo_finalistas.png`
