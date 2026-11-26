# Proyecto final

## Introducción
Explicar el problema

## Motivación
Por qué es importante

## Revisión bibliográfica

## Diagrama esquemático de la propuesta

### Decisiones tomadas

- **Variable objetivo:** `NObeyesdad` con sus **7 clases** tal como vienen en el dataset (`Insufficient_Weight`, `Normal_Weight`, `Overweight_Level_I`, `Overweight_Level_II`, `Obesity_Type_I`, `Obesity_Type_II`, `Obesity_Type_III`). No se agrupan.
- **Dataset:** se usa el CSV original de [UCI](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition). El CSV actual del repositorio tiene los decimales dañados (ej. `1.616.366` en vez de `1.616366`) y no se puede corregir de forma confiable, porque la posición del decimal depende de cada columna.

### En discusión

- **Variables de entrada:** como la clase se calcula con el IMC (`peso / altura²`), incluir `Height` y `Weight` produce fuga de datos (data leakage). Opciones:
  - A. Solo hábitos.
  - B. Todas las variables.
  - C. Solo hábitos como modelo principal y todas las variables como referencia.
- **Pipeline, modelos, métricas y uso de PCA:** pendientes.

## Explicación del dataset
poner el link del cuaderno
