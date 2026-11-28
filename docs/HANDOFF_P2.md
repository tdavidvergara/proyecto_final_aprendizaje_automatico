# Handoff · P2 (Ingeniería de Hábitos + Yeo-Johnson + StandardScaler)

Documento para el agente que continúa este trabajo. Léelo completo antes de tocar el repositorio.

## 1. Contexto del proyecto

- **Curso:** Aprendizaje Automático (UAO). **Entrega:** sábado 2026-11-28.
- **Problema:** clasificar los 7 niveles de obesidad (`NObeyesdad`) del dataset UCI *Estimation of obesity levels based on eating habits and physical condition* (id 544, 2,111 filas, ~77 % sintéticas por SMOTE).
- **Equipo:** Sebastián Giraldo, Violeta España, Luis Murillo, Juan David Velasco y David Vergara.
- **Qué pide la guía** (`course_project_guide.pdf`): paper IEEE en inglés con **al menos 3 metodologías de preprocesamiento** y **al menos 3 modelos con análisis de complejidad**, más el despliegue. El equipo lo organiza como una matriz 3×3.

| Metodología | Qué hace | Responsable |
|---|---|---|
| P1 | 17 variables base + RobustScaler | Sebastián |
| **P2** | 4 variables de hábitos + Yeo-Johnson + StandardScaler (21 variables) | **Luis (hecho, ver abajo)** |
| P3 | RobustScaler + PCA 95 % (~11 componentes) | Compañero/a (sin confirmar) |

| Modelo | Familia | Responsable |
|---|---|---|
| M1 | Random Forest | Sebastián |
| M2 | XGBoost | Compañera |
| M3 | MLP (`MLPClassifier`) | Sin confirmar |

Primero cada modelo se prueba con una metodología ("uno con uno") y después se cruzan todos. Cada persona trabaja en su rama y hace merge a `develop`. La base es `main`.

## 2. Contrato de datos (obligatorio para que la matriz sea comparable)

Igual que `Desarrollo.ipynb`, sin cambios:

1. Cargar desde UCI con `ucimlrepo.fetch_ucirepo(id=544)` y armar `dataset = pd.concat([X, y], axis=1)`. **No usar** `dataset/ObesityDataSet_raw_and_data_sinthetic.csv`: tiene los decimales dañados.
2. Codificación (celda 32): binarias a 0/1; `FCVC`, `NCP`, `CH2O`, `FAF` y `TUE` redondeadas a entero; `CAEC`/`CALC` ordinales (con `CALC` "Always" → "Frequently"); `MTRANS` one-hot (Bike y Motorbike → `Two_Wheels`); clase de 0 a 6.
3. Eliminar duplicados **antes** de dividir: 2,111 → 2,087.
4. `train_test_split(y.index, test_size=0.2, stratify=y, random_state=20)` → entrenamiento 1,669 × 17, prueba 418 × 17.
5. Benchmark **preventivo** `X_prev` (17 columnas, sin `Weight`, `Height` ni `IMC`) como principal; **operativo** `X_oper` solo como referencia.
6. Todo preprocesamiento va **dentro de un `Pipeline`** y se ajusta solo con entrenamiento (o dentro de cada pliegue de validación cruzada).

## 3. Lo que se hizo (rama `feat/p2-habitos`)

| Commit | Contenido |
|---|---|
| `e635556` | `src/p2_habitos.py` + `tests/test_p2_habitos.py` (9 tests) |
| `6b23cbe` | `P2_habitos.ipynb` ejecutado (14/14 celdas, sin errores), con conclusiones en la sección 9 |

**Uso de P2 en cualquier modelo:**

```python
from sklearn.pipeline import Pipeline
from src.p2_habitos import construir_p2

modelo = Pipeline([("p2", construir_p2(X_train)), ("modelo", XGBClassifier(...))])
modelo.fit(X_train, y_train)
```

**Las 4 variables nuevas** (fila a fila, sin parámetros aprendidos):

- `Balance_Act_Sed = FAF / (TUE + 1)`
- `Hidratacion_Activa = CH2O * (FAF + 1)`, que **nunca** se divide por el peso
- `Riesgo_Gen_Cal = family_history_with_overweight * FAVC`
- `Indice_Protector = FCVC + CH2O + FAF + SCC - FAVC`, con **signos supuestos** (la fórmula original llegó cortada)

Yeo-Johnson (`standardize=True`) se aplica a las 11 columnas no binarias; las 10 binarias o one-hot solo pasan por `StandardScaler`. `agregar_habitos` lanza `ValueError` si recibe `Weight`, `Height` o `IMC`.

## 4. Resultados verificados

- **Fuga:** P2 no filtra la clase. El contraejemplo `CH2O / Weight` sí filtra: XGBoost pasa de 0.801 a **0.890** de F1-macro en prueba y la correlación de Spearman con la clase es −0.53. Por eso esa variable solo puede ir en el benchmark operativo.
- **IMC predicho con regresión lineal** (R² en prueba): 0.411 con las 17 originales y 0.439 con las 21 de P2. Se debe a la no linealidad de las variables nuevas, no a información nueva.
- **Asimetría antes → después de Yeo-Johnson:** `CAEC` 2.06 → −0.16, `Age` 1.47 → 0.09, `NCP` −1.09 → −0.09.
- **F1-macro P1 vs. P2** (validación cruzada de 5 pliegues solo sobre entrenamiento, hiperparámetros por defecto, sin ajuste):

| Modelo | P1 | P2 |
|---|---|---|
| Random Forest | 0.795 ± 0.015 | 0.777 ± 0.009 |
| XGBoost | 0.780 ± 0.014 | 0.789 ± 0.014 |
| MLP | 0.509 ± 0.072 | **0.565 ± 0.031** |

## 5. Revisión de código (Gentle AI)

El commit `e635556` fue **aprobado** con la perspectiva de confiabilidad. Dejó hallazgos no bloqueantes, todos como trabajo posterior:

| Id | Severidad | Dónde | Qué hacer |
|---|---|---|---|
| R3-001 | Warning | `tests/test_p2_habitos.py:110-115` | El test de "no reajusta con prueba" no detecta fuga: `transform` nunca reajusta. Verificar la salida de `transform(X_test)`: 21 columnas, nombres iguales a `get_feature_names_out()` y valores finitos. |
| R3-002 | Warning | `src/p2_habitos.py:56-58` | La división continua/binaria usa `nunique() > 2` sobre `X_train_ejemplo`. Si una muestra chica deja una columna continua con 2 o menos valores, va a StandardScaler sin avisar. Conviene fijar la lista de columnas continuas de forma explícita. |
| R3-003 | Warning | `src/p2_habitos.py:38-48` | `agregar_habitos` no valida el esquema de entrada. Una columna de más (por ejemplo la clase) pasa sin error, y una columna faltante da un `KeyError` poco claro. Validar que entren exactamente las 17 columnas esperadas. |
| R3-004 | Sugerencia | `src/p2_habitos.py:32-34` | El test de `Indice_Protector` prueba la fórmula supuesta, no la correcta. Queda abierto hasta confirmarla. |
| R3-005 | Sugerencia | `src/p2_habitos.py:44-47` | No hay validación de tipos ni de rangos: `TUE = -1` daría infinito, y los NaN se propagan. |

El notebook (`6b23cbe`) no se revisó: casi todo son salidas generadas y excede el límite de contexto del revisor.

## 6. Pendientes, en orden de prioridad

1. **Confirmar con Sebastián los signos de `Indice_Protector`.** Evaluar quitarla: es redundante (combinación lineal exacta de 5 columnas) y su correlación con la clase no es significativa (−0.025, p = 0.31).
2. **Abrir el PR** de `feat/p2-habitos` hacia `develop` (la rama ya está subida).
3. Corregir los hallazgos R3-001 a R3-003 si queda tiempo antes del sábado.
4. Verificar que los notebooks de M1, M2 y M3 usen el mismo contrato de datos de la sección 2 y que importen P2 desde `src/p2_habitos.py`, en lugar de copiar el código.
5. **Pendientes del informe** `Informe_Estado_del_Arte_Obesidad_ML_DL.md` (revisado el 2026-11-27):
   - La sección 3 propone stacking con LightGBM, CatBoost y TabNet, y la sección 4 usa regresión logística, Random Forest, XGBoost y MLP. Hay que alinearlas.
   - La comparación con Kaggle s4e2 no es válida: es otro dataset (~20 mil filas), hubo 3,587 equipos (el informe dice "más de 3,700") y el rango 91.5–92.2 % no tiene fuente.
   - La tabla cita "De-La-Hoz-Franco et al. (2021)", que no está en la bibliografía.
   - Los subniveles de sobrepeso (corte en 27.5) no son de la OMS; los definieron los autores del dataset.
   - Falta decidir qué hacer con `Gender`. Recomendación: comparar con y sin la variable y evaluar también solo sobre las ~500 filas reales.

## 7. Cómo reproducir

```bash
pip install scikit-learn pandas numpy scipy xgboost ucimlrepo matplotlib seaborn pytest nbconvert ipykernel
python -m pytest tests -q            # desde la raíz del repo: 9 passed
jupyter nbconvert --to notebook --execute --inplace P2_habitos.ipynb   # necesita internet (UCI)
```

No hay `requirements.txt` en el repo.

## 8. Reglas de trabajo

- No modificar archivos de los compañeros (`Desarrollo.ipynb`, el informe, sus notebooks). Proponer cambios en una rama propia.
- Commits en formato *Conventional Commits*, sin líneas de coautoría de IA.
- Push, PR y merge los decide Luis.
- Revisiones de Gentle AI: solo para cambios de riesgo medio tirando a alto o alto.
- No subir datasets ni modelos (`.pkl`) al repo.
