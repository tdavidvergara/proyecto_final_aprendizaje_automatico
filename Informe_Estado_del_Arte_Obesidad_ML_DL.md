# PROYECTO FINAL: ESTIMACIÓN DE NIVELES DE OBESIDAD MEDIANTE MACHINE LEARNING Y DEEP LEARNING
**Integrantes:** Sebastián Giraldo, Violeta España, Luis Murillo, Juan David Velasco & David Vergara  
**Dataset Oficial:** *Estimation of obesity levels based on eating habits and physical condition* (UCI Machine Learning Repository / Kaggle Playground Series s4e2)  
**Fecha:** Octubre de 2026  

## 1. INTRODUCCIÓN: DESCRIPCIÓN DEL PROBLEMA

La obesidad representa uno de los principales desafíos de salud pública a nivel mundial debido a su relación con diversas enfermedades crónicas y sus implicaciones en la calidad de vida de las personas. Su desarrollo es un fenómeno multifactorial en el que intervienen diferentes elementos, entre ellos los hábitos alimenticios, la actividad física, los antecedentes familiares, las condiciones metabólicas y los comportamientos asociados al estilo de vida.

Tradicionalmente, la evaluación del estado nutricional se ha apoyado en indicadores antropométricos como el Índice de Masa Corporal (IMC), calculado a partir de la relación entre el peso y la estatura de una persona. Este indicador permite establecer categorías generales de clasificación del peso corporal. Sin embargo, el estudio de la obesidad requiere considerar diferentes factores que pueden estar relacionados con su aparición y evolución, especialmente cuando se busca comprender la influencia de los hábitos y las condiciones individuales.

En este contexto, surge la necesidad de explorar herramientas computacionales que permitan analizar grandes cantidades de información, identificar patrones y establecer relaciones entre diferentes características de los individuos. El aprendizaje automático (Machine Learning) ofrece alternativas para abordar este tipo de problemas mediante algoritmos capaces de aprender a partir de datos históricos y generar predicciones sobre nuevas observaciones.

El problema específico que aborda este proyecto consiste en **determinar si es posible estimar los niveles de obesidad de una persona a partir de sus características físicas, antecedentes familiares, hábitos alimenticios y comportamientos relacionados con su estilo de vida, utilizando modelos de aprendizaje automático y aprendizaje profundo**.

Para desarrollar esta investigación se utilizará el conjunto de datos denominado *Estimation of Obesity Levels Based on Eating Habits and Physical Condition*, disponible en el repositorio de la Universidad de California en Irvine (UCI Machine Learning Repository). Este conjunto de datos contiene 2.111 registros de individuos provenientes de Colombia, México y Perú, con información relacionada con sus características demográficas, condiciones físicas y hábitos cotidianos.

El conjunto está compuesto por 16 variables predictoras y una variable objetivo denominada `NObeyesdad`, encargada de representar siete categorías:

- **Insufficient Weight:** Bajo peso.
- **Normal Weight:** Peso normal.
- **Overweight Level I:** Sobrepeso grado I.
- **Overweight Level II:** Sobrepeso grado II.
- **Obesity Type I:** Obesidad tipo I.
- **Obesity Type II:** Obesidad tipo II.
- **Obesity Type III:** Obesidad tipo III.

Las variables predictoras incluyen características como edad, género, estatura, peso y antecedentes familiares de sobrepeso. También contemplan información sobre el consumo de alimentos con alto contenido calórico, frecuencia de consumo de vegetales, número de comidas diarias, consumo de agua, actividad física, tiempo frente a dispositivos electrónicos y otros comportamientos relacionados con el estilo de vida.

A partir de esta información, el proyecto plantea un problema de clasificación supervisada multiclase, en el cual los modelos deberán aprender las relaciones existentes entre las variables de entrada y las siete categorías de la variable objetivo.

Es importante considerar que el conjunto de datos contiene registros originales y registros generados sintéticamente mediante técnicas de sobremuestreo. Por esta razón, será necesario implementar estrategias de entrenamiento y validación que eviten la fuga de información (*Data Leakage*) y permitan obtener resultados experimentales confiables.

De esta manera, el proyecto busca desarrollar una solución computacional que permita estudiar la capacidad predictiva de diferentes algoritmos y comprender qué características tienen mayor relación con los niveles de obesidad.

## 2. MOTIVACIÓN: IMPORTANCIA DEL PROBLEMA

La importancia de este proyecto se fundamenta en la necesidad de comprender mejor los factores asociados con la obesidad y explorar cómo las herramientas de inteligencia artificial pueden contribuir al análisis de información relacionada con la salud.

La obesidad no depende exclusivamente de una característica física. Por el contrario, está relacionada con múltiples factores que interactúan entre sí, como la alimentación, la actividad física, los antecedentes familiares y las condiciones individuales. Esta complejidad hace que el análisis conjunto de diferentes variables sea especialmente relevante para identificar patrones y estudiar posibles relaciones entre los comportamientos cotidianos y las categorías de peso corporal.

Desde una perspectiva de salud pública, identificar estos patrones puede contribuir a comprender mejor los factores asociados con el desarrollo de la obesidad y aportar información que sirva como apoyo en el diseño de estrategias de educación, prevención y promoción de hábitos saludables.

### 2.1. Importancia desde la perspectiva de la salud pública

La obesidad está asociada con un mayor riesgo de desarrollar diferentes enfermedades crónicas, entre ellas la diabetes tipo 2, las enfermedades cardiovasculares y algunos trastornos metabólicos.

Por esta razón, comprender los factores relacionados con los diferentes niveles de obesidad es importante para fortalecer las iniciativas orientadas a la promoción de estilos de vida saludables.

El análisis de variables como la frecuencia de actividad física, los hábitos alimenticios, el consumo de agua y el tiempo dedicado a actividades sedentarias permite estudiar cómo estos comportamientos se relacionan con las categorías de obesidad presentes en el conjunto de datos.

De esta manera, el proyecto busca aportar una perspectiva analítica que permita comprender mejor las relaciones entre los hábitos cotidianos y las condiciones físicas de los individuos.

### 2.2. Importancia desde la perspectiva tecnológica

El desarrollo de este proyecto también representa una oportunidad para aplicar técnicas de inteligencia artificial a un problema real de interés social.

Los algoritmos de Machine Learning permiten procesar múltiples variables simultáneamente, identificar relaciones no lineales y construir modelos de clasificación a partir de patrones presentes en los datos.

A su vez, las técnicas de Deep Learning permiten explorar arquitecturas neuronales capaces de aprender representaciones más complejas de la información.

La comparación entre ambos enfoques permitirá estudiar sus ventajas, limitaciones y capacidad de generalización en un problema de clasificación multiclase.

Además, el uso de técnicas de aprendizaje por ensamble permitirá investigar si la combinación de diferentes modelos puede aprovechar sus capacidades individuales y mejorar el rendimiento de las predicciones.

### 2.3. Importancia del análisis de los hábitos y el estilo de vida

Uno de los aspectos más relevantes de esta investigación es estudiar la contribución de los hábitos alimenticios y los comportamientos cotidianos en la estimación de los niveles de obesidad.

Aunque variables como el peso y la estatura ofrecen información directa sobre la condición física de una persona, resulta interesante investigar qué capacidad predictiva tienen otras características, como:

- Frecuencia de consumo de alimentos con alto contenido calórico.
- Consumo habitual de vegetales.
- Número de comidas realizadas durante el día.
- Consumo diario de agua.
- Frecuencia de actividad física.
- Tiempo dedicado a dispositivos electrónicos.
- Hábitos de consumo de alcohol y tabaco.
- Antecedentes familiares de sobrepeso.

### 2.4. Importancia académica y aporte del proyecto

Desde el ámbito académico, esta investigación permite integrar conocimientos de estadística, análisis exploratorio de datos, procesamiento de información, clasificación supervisada, optimización de modelos y evaluación de algoritmos.

Asimismo, ofrece la oportunidad de realizar una comparación experimental entre diferentes paradigmas de aprendizaje automático, utilizando métricas como Accuracy, Precision, Recall y F1-Score, con especial atención al desempeño de cada modelo en las siete categorías.

Otro aspecto importante será la interpretación de los resultados mediante técnicas de explicabilidad, que permitan identificar cuáles variables tienen mayor influencia en las predicciones y comprender mejor el comportamiento de los modelos.

El proyecto también busca destacar la importancia de una metodología experimental rigurosa. No se trata únicamente de conseguir un porcentaje elevado de exactitud, sino de garantizar que los resultados sean obtenidos mediante una validación adecuada, sin contaminación entre los conjuntos de entrenamiento y evaluación.

### 2.5. Propósito final

En conjunto, esta investigación busca demostrar cómo las técnicas de Machine Learning y Deep Learning pueden utilizarse para analizar información relacionada con los hábitos alimenticios, las condiciones físicas y los niveles de obesidad.

Más allá de desarrollar un modelo con un buen rendimiento predictivo, se pretende comprender las capacidades y limitaciones de los diferentes algoritmos, identificar las variables más relevantes y evaluar la influencia de las características biométricas y los hábitos de vida en el proceso de clasificación.

El propósito es construir una solución experimental fundamentada en datos, que permita explorar el potencial de la inteligencia artificial en problemas relacionados con la salud y generar conocimiento sobre las relaciones existentes entre el estilo de vida y las categorías de obesidad.

**Nota sobre el alcance:** Los modelos desarrollados tendrán un propósito académico y experimental. Sus resultados no deben interpretarse como diagnósticos médicos ni como herramientas clínicas validadas para tomar decisiones sobre la salud de una persona.


# 3. INFORME DEL ESTADO DEL ARTE Y PROPUESTA DE SOLUCIÓN: ESTIMACIÓN DE NIVELES DE OBESIDAD MEDIANTE APRENDIZAJE SUPERVISADO CLÁSICO (MACHINE LEARNING) Y PROFUNDO (DEEP LEARNING)

---

## 3.1. RESUMEN EJECUTIVO

El presente informe consolida la revisión crítica del Estado del Arte y expone de manera formal la Propuesta de Solución Metodológica y Algorítmica para el problema de clasificación multiclase de 7 niveles de obesidad a partir del conjunto de datos de Palechor y de la Hoz Manotas (UCI Machine Learning Repository, N \= 2,111).

Tras una auditoría técnica rigurosa de la literatura —la cual desmitifica las precisiones artificialmente infladas del 97% al 98% reportadas en trabajos pioneros debido a sobremuestreo sintético (SMOTE) aplicado previo a la partición de validación cruzada (fuga de datos o data leakage)—, se establece la verdadera cota técnica del Estado del Arte: un rendimiento reproducible y blindado situado entre el 91.5% y el 92.2% de exactitud multiclase (Accuracy).

Como aporte sustancial e innovador de este proyecto, se diseña y formula una Propuesta de Solución Híbrida Supervisada que contrasta y articula dos grandes paradigmas de aprendizaje:

1. Rama de Machine Learning Clásico (Tree-based Gradient Boosting): Implementación de LightGBM, XGBoost y CatBoost, optimizados para modelar particiones ortogonales no lineales en datos tabulares mixtos.  
2. Rama de Deep Learning Tabular (Redes Neuronales Profundas): Implementación de TabNet (arquitectura con mecanismos de atención dispersa secuencial Sparsemax para selección dinámica de variables paso a paso) y un Perceptrón Multicapa Regularizado (Tabular MLP con Entity Embeddings) para aprender representaciones densas y no lineales de hábitos cualitativos.  
3. Meta-Ensamble por Apilamiento (Stacking Ensemble): Fusión de probabilidades Out-Of-Fold (OOF) de ambas ramas mediante un meta-clasificador regularizado, maximizando el F1-Score macro y resolviendo la confusión en fronteras clínicas adyacentes (Sobrepeso I vs. II y Sobrepeso II vs. Obesidad I).  
4. Diseño de Validación con Benchmark Dual: Comparación formal de un modelo operativo con variables biométricas directas frente a un modelo clínico preventivo sustentado exclusivamente en hábitos de nutrición y actividad física, neutralizando el fenómeno de Target Leakage del Índice de Masa Corporal.

---

## 3.2. CONTEXTUALIZACIÓN DEL PROBLEMA Y DATASET OFICIAL

### 3.2.1. Naturaleza y Estructura del Conjunto de Datos

El dataset oficial comprende 2,111 registros y 17 atributos (16 variables predictoras continuas, ordinales y nominales, más 1 variable objetivo multiclase NObeyesdad).

Las variables se dividen en tres dimensiones fundamentales:

1. Biométricas y Genéticas: Género (Gender), Edad (Age), Estatura (Height), Peso (Weight) y Antecedentes familiares con sobrepeso (family\_history\_with\_overweight).  
2. Hábitos de Alimentación: Consumo frecuente de alimentos hipercalóricos (FAVC), Consumo de vegetales (FCVC), Número de comidas principales (NCP), Picoteo entre comidas (CAEC), Consumo de agua diario (CH2O) y Monitoreo de calorías (SCC).  
3. Actividad Física y Estilo de Vida: Tabaquismo (SMOKE), Frecuencia de actividad física (FAF), Tiempo frente a pantallas (TUE), Consumo de alcohol (CALC) y Medio de transporte usual (MTRANS).

### 3.2.2. Las 7 Categorías Clínicas Objetivo (NObeyesdad)

La variable dependiente categoriza a los individuos de acuerdo con los umbrales estándar de la OMS:

1. Insufficient\_Weight (Bajo peso: IMC menor a 18.5)  
2. Normal\_Weight (Peso normal: 18.5 ≤ IMC \< 25.0)  
3. Overweight\_Level\_I (Sobrepeso Grado I: 25.0 ≤ IMC \< 27.5)  
4. Overweight\_Level\_II (Sobrepeso Grado II: 27.5 ≤ IMC \< 30.0)  
5. Obesity\_Type\_I (Obesidad Tipo I: 30.0 ≤ IMC \< 35.0)  
6. Obesity\_Type\_II (Obesidad Tipo II: 35.0 ≤ IMC \< 40.0)  
7. Obesity\_Type\_III (Obesidad Grado III / Mórbida: IMC ≥ 40.0)

---

## 3.3. AUDITORÍA CRÍTICA DE LA LITERATURA PREVIA

### 3.3.1. Génesis del Dataset y la Fuga de Datos por SMOTE Pre-Split

En el artículo original (Data in Brief, 2019), Palechor y de la Hoz Manotas recolectaron 498 encuestas reales (23.6%) en Colombia, México y Perú. Para corregir el desbalance de clases nativo, aplicaron en la herramienta Weka el algoritmo SMOTE, sintetizando 1,613 registros adicionales hasta totalizar 2,111 observaciones.

El algoritmo SMOTE interpola linealmente entre vecinos más cercanos (k-NN) mediante la siguiente relación:

Nuevo\_Dato \= Dato\_i \+ Factor\_Aleatorio \* (Vecino\_cercano \- Dato\_i)

Al aplicar SMOTE globalmente sobre la matriz antes de particionar entre conjunto de entrenamiento y conjunto de prueba:

1. Puntos sintéticos en el conjunto de prueba son clones geométricos directos de instancias en el conjunto de entrenamiento.  
2. Los modelos memorizan la vecindad euclidiana, arrojando precisiones artificiales del 97% al 98.5%.  
3. Además, SMOTE generó decimales continuos irreales sobre variables que originalmente eran escalas enteras o Likert discretas en la encuesta (por ejemplo, FCVC \= 2.418, FAF \= 0.573), actuando como huellas digitales estocásticas (fingerprints) que facilitaron la memorización algorítmica.

### 3.3.2. La Frontera SOTA Real: Kaggle Playground Series s4e2 (2024)

En la competencia internacional Kaggle s4e2 (más de 3,700 participantes evaluados sobre conjuntos de prueba ciegos hold-out sin fuga de datos), la cota máxima demostrada se sitúa estrictamente en 91.5% a 92.2% de Accuracy. Cualquier resultado superior en este dominio con datos comparables evidencia sobreajuste o fuga metodológica.

---

## 3.4. PROPUESTA FORMAL DE SOLUCIÓN: ARQUITECTURA HÍBRIDA SUPERVISADA (ML \+ DL)

Para responder a las exigencias académicas y construir una solución técnica de alto impacto, se propone una arquitectura modular de dos ramas supervisadas concurrentes:

1. Ingesta y Preprocesamiento In-Fold:  
   - Partición mediante Validación Cruzada Estratificada de 5 Pliegues (Stratified 5-Fold).  
   - Escalado robusto con RobustScaler y codificación ordinal de variables jerárquicas (CAEC y CALC: no \= 0, Sometimes \= 1, Frequently \= 2, Always \= 3).  
2. Ingeniería de Características Biofísicas:  
   - BMI Canónico \= Peso / (Estatura^2)  
   - Ratio de Hidratación por Biomasa \= CH2O / Peso  
   - Balance Actividad Física vs. Sedentarismo \= FAF / (TUE \+ 1.0)  
3. Rama Machine Learning Clásico (GBDT):  
   - LightGBM (Crecimiento Leaf-wise y algoritmo GOSS).  
   - XGBoost (Histograma regularizado con penalizaciones L1 y L2).  
   - CatBoost (Ordered Boosting y árboles simétricos).  
4. Rama Deep Learning Tabular:  
   - TabNet (Atención dispersa secuencial con Sparsemax).  
   - Tabular MLP (Perceptrón Multicapa con capas de Entity Embeddings, Batch Normalization y Dropout).  
5. Meta-Ensamble por Apilamiento (Stacking de Nivel 2):  
   - Extracción de probabilidades cruzadas Out-Of-Fold (OOF) de las 7 clases para los 5 modelos (Matriz Z de dimensión N x 35).  
   - Meta-clasificador lineal penalizado (Ridge / Regresión Logística L2).  
   - Post-procesamiento mediante optimización Simplex Nelder-Mead para calibrar multiplicadores de frontera y resolver la confusión entre clases adyacentes (Overweight I vs. II y Overweight II vs. Obesity I).  
6. Validación con Benchmark Dual:  
   - Benchmark A (Operativo): Incluye peso, altura e IMC calculado (Meta: 91.5% \- 92.2% de Accuracy).  
   - Benchmark B (Clínico-Preventivo): Excluye peso y altura para medir la capacidad predictiva pura de los hábitos de vida sin Target Leakage (Rendimiento esperado: 80% \- 82%).

---

## 3.5. DESGLOSE TÉCNICO DE LOS MODELOS SUPERVISADOS

### 3.5.1. Rama Machine Learning: Ensambles de Árboles (GBDT)

A. LightGBM (Light Gradient Boosting Machine)

- Crecimiento de árboles guiado por hojas (leaf-wise) con restricción de profundidad máxima.  
- Muestreo unilateral por gradientes (GOSS), conservando instancias con altos gradientes de error y muestreando las de bajo gradiente para acelerar el cómputo sin perder precisión.  
- Minimización de la entropía cruzada multiclase en las 7 categorías ponderales.

B. XGBoost (Extreme Gradient Boosting)

- Optimización analítica de segundo orden mediante expansión de Taylor.  
- Regularización estructural explícita L1 (alpha) y L2 (lambda) sobre los pesos de las hojas para mitigar la varianza inducida por los datos sintetizados con SMOTE.  
- Particionamiento acelerado por histogramas continuos.

C. CatBoost (Categorical Boosting)

- Algoritmo de Ordered Boosting que previene el sesgo de estimación del gradiente en conjuntos tabulares de tamaño mediano.  
- Árboles simétricos (oblivious trees) donde la misma regla de decisión se aplica a todo un nivel de profundidad, actuando como un regularizador implícito frente a patrones ruidosos.  
- Codificación nativa de variables categóricas mediante Target Statistics dinámicas.

### 3.5.2. Rama Deep Learning: Arquitecturas Neuronales para Datos Tabulares

A. TabNet (Google Cloud AI)

- Diseñado para superar a las redes neuronales estándar en datos tabulares combinando aprendizaje profundo con la interpretabilidad de los árboles de decisión.  
- Mecanismo de Atención Dispersa Secuencial (Sparsemax): En cada paso de decisión s, un transformador de atención genera una máscara dinámica que selecciona qué variables inspeccionar, penalizando el reuso excesivo de factores ya evaluados.  
- Bloques de procesamiento con Gated Linear Units (GLU) y conexiones residuales normalizadas.  
- Interpretabilidad Nativa: Entrega matrices de asignación de atención que permiten identificar visualmente qué variables determinaron la predicción de cada paciente.

B. Tabular MLP con Entity Embeddings

- Capas de Embeddings: Variables nominales (MTRANS) y ordinales (CAEC, CALC) se proyectan a vectores continuos densos, permitiendo que la red aprenda relaciones geométricas de proximidad entre hábitos.  
- Bloques Densos Residuales: 3 capas ocultas (128, 64 y 32 neuronas) con activaciones LeakyReLU / Mish, capas de Batch Normalization y Dropout (0.25) para prevenir la coadaptación de neuronas.  
- Optimizador AdamW con decaimiento de pesos y programador de tasa de aprendizaje por recocido de coseno (Cosine Annealing).

---

## 3.6. MATRIZ COMPARATIVA DEL ESTADO DEL ARTE Y NUESTRA PROPUESTA

| Estudio / Referencia | Modelos Evaluados | Paradigma | Validación | Métricas Reportadas | Diagnóstico Metodológico |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Palechor & de la Hoz (2019) | Decision Tree (J48), Naïve Bayes, Simple Logistic | ML Clásico | Weka 10-Fold CV sobre dataset inflado | Precision: 97.4%, Accuracy: 97.5% | Fuga de datos severa: SMOTE pre-split y decimales espurios. |
| De-La-Hoz-Franco et al. (2021) | Random Forest, Extra Trees, XGBoost | ML Clásico | CV estándar sobre los 2,111 registros | Accuracy: 98.1% (Random Forest) | Fuga heredada del dataset original; no corrige el sobreajuste sintético. |
| Yağın et al. (2023) | Red Neuronal Artificial (ANN / MLP) | Deep Learning | Normalización MinMax \+ Optimización Bayesiana | Accuracy: 93.06%, F1-Score: 92.79% | Buen ajuste bayesiano de capas densas, pero incluye peso/altura. |
| Murickan & Toma (2024) | Random Forest, Gradient Boosting | ML Clásico | Ablación: Con Peso/Altura vs. Sin Peso/Altura | Sin Peso: 81.65% Accuracy / Con Peso: 97.5% | Metodología impecable: Demuestra el Target Leakage del IMC. |
| Kaggle s4e2 Top 1% (2024) | Ensamble Stacking (LGBM, XGBoost, CatBoost, TabNet) | Híbrido ML \+ DL | Stratified 10-Fold CV ciego sobre hold-out privado | Accuracy: 91.9% \- 92.2% | Techo técnico real comprobado en entorno competitivo sin fuga. |
| NUESTRA PROPUESTA | Híbrido ML \+ DL con Stacking: LightGBM, XGBoost, CatBoost, TabNet y MLP | Híbrido Multimodelo | Benchmark Dual: Operativo SOTA vs. Causal sin Peso/Altura | Proyectado Bench A: 91.5% \- 92.2% / Bench B: 80% \- 82% | Diseño óptimo: Cero fuga de datos, contraste ML vs. DL y explicabilidad con SHAP. |

---

## 3.7. REFERENCIAS BIBLIOGRÁFICAS

1. Palechor, F. M., & de la Hoz Manotas, A. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in Colombia, Peru and Mexico. Data in Brief, 25, 104344\.  
2. De-La-Hoz-Correa, E., Mendoza-Palechor, F., De-La-Hoz-Manotas, A., Morales-Ortega, R., & Sánchez-Bohórquez, A. (2019). Obesity Level Estimation Software based on Decision Trees. Journal of Computer Science, 15(1), 67-77.  
3. Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. Proceedings of the AAAI Conference on Artificial Intelligence, 35(8), 6679–6687.  
4. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 3146–3154.  
5. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.  
6. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. Advances in Neural Information Processing Systems (NeurIPS 2018), 31, 6638–6648.  
7. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 4765–4774.  
8. Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2), 241-259.  
9. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321-357.  
10. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? Advances in Neural Information Processing Systems (NeurIPS 2022), 35, 507–520.  
11. Yağın, F. H., Gülü, M., Görmez, Y., et al. (2023). Estimation of Obesity Levels with a Trained Neural Network Approach optimized by the Bayesian Technique. Applied Sciences, 13(6), 3866\.  
12. Murickan, A., & Toma, M. (2024). Behavioral Lifestyle Factors Versus Medical History in Determining the Predictive Power of Machine Learning-Based Obesity Classification. Technologies, 12(5), 64\.  
13. Kaggle. (2024). Multi-Class Prediction of Obesity Risk. Kaggle Playground Series \- Season 4, Episode 2\.  
14. World Health Organization (WHO). (2024). Obesity and overweight fact sheet. WHO Media Centre, Geneva.
