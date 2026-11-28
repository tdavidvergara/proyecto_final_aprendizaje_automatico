# Proyecto final Aprendizaje Automático
## Integrantes: 
**Sebastián Giraldo, Violeta España, Luis Murillo, Juan David Velasco & David Vergara**

We live in a world flooded with data. Globalization, widespread internet and mobile applications have converted us into a continuous source of information for big corporations, and also has given us the opportunity to access knowledge in a way that had never been possible before. What can you use artificial intelligence for? the answer is: pretty much the same that you can use your intelligence for.  So, the question would be: what do YOU want to use AI for? In this project you will find a practical application of machine learning, something that motivates you and is relevant to people out there. Then, you will use the tools learnt during our course and provide a solution that satisfies both your appetite for knowledge and the needs of the potential users. There will be two deliverables, the first one will be a “project proposal” which will be delivered on the 8th session. The second will be a scientific paper, in which you will communicate effectively the results of your research. 

## First deliverable: 
**Project Proposal You will make a presentation with the following sections:** 
1. Introduction and motivation:
   ● What is the problem (application) you will tackle?
   ● Why is it important?
2. Literature review about other solutions to the problem
3. Your solution/process description, including an schematic diagram
4. Dataset explanation and visualization You will submit the slides of your presentation. You will have 15 minutes to present and 5 mins for questions.  

## App de escritorio

Aplicación de escritorio (CustomTkinter) que usa el modelo ganador (`modelos/modelo_ganador.pkl`, XGBoost con preprocesamiento P2) para estimar el nivel de obesidad a partir de hábitos, antecedentes, edad y género. Muestra la clase predicha, la probabilidad de cada una de las 7 clases y las métricas del modelo (`modelos/metricas_ganador.json`).

> Modelo académico con fines educativos: no constituye un diagnóstico médico.

### Requisitos

- Windows 10/11 (también funciona en otros sistemas con Tk disponible).
- [uv](https://docs.astral.sh/uv/) instalado. uv descarga Python 3.13 si no está disponible.

### Uso

```bash
uv sync          # crea .venv e instala las dependencias fijadas en uv.lock
uv run app       # abre la ventana "Clasificador de Obesidad"
uv run pytest    # ejecuta las pruebas
```

### Notas

- `scikit-learn==1.6.1` y `xgboost==3.4.1` están fijados a las versiones con las que `Notebook_Matriz.ipynb` guardó el modelo.
- El modelo se guardó en Linux (Colab). `app_obesidad/core.py` reconstruye el booster de XGBoost a partir de su sección portable "Model" para que cargue en Windows.
- El instalador (.exe con PyInstaller / Inno Setup) está fuera del alcance por ahora.
