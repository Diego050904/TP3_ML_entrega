# TP3 – Competencia de ML: predicción de cancelaciones de reservas de GPU

72.75 Aprendizaje Automático (Machine Learning) · ITBA · 2026 Q2

**Integrantes:**
- Diego Emiliano Godinez Ortega
- Thomas Michael Spitzner
- Guillermo Arturo Ledesma Campos


## Estructura

```
├── predictor.py                  función predictor (entregable)
├── cancellation_model.joblib     modelo entrenado (entregable, lo genera el notebook 7)
├── data/
│   ├── gpu_resource_reservations.csv   dataset original
│   └── train.csv, test.csv, *.json     archivos generados por los notebooks
├── notebooks/                    ejecutar en orden
├── figures/                      gráficas para la presentación
├── requirements.txt
└── README.md
```

## Notebooks

| Notebook | Contenido |
|---|---|
| `01_cleaning_split.ipynb` | Leakage (`execution_status`, `execution_status_date`), faltantes, consistencia, fuera de dominio, duplicados, split 80/20 estratificado |
| `02_eda.ipynb` | EDA sobre train |
| `03_feature_engineering.ipynb` | Variables nuevas, información mutua, PCA + K-means |
| `04_model_comparison.ipynb` | 8 clasificadores con 5-fold CV estratificado |
| `05_tuning_diagnostics.ipynb` | Optuna sobre LightGBM, curva de validación, curva de aprendizaje, Group K-Fold |
| `06_ensemble_threshold.ipynb` | Promedio y stacking de LightGBM, XGBoost y CatBoost; umbral óptimo para balanced accuracy |
| `07_final_model.ipynb` | Evaluación única en test, análisis de errores, reentrenamiento con todos los datos y verificación del predictor |

## Resultados

Modelo final: **LightGBM** (Optuna, 30 trials), con umbral de decisión 0.41 elegido para maximizar la balanced accuracy.

| Evaluación | Balanced accuracy | ROC-AUC |
|---|---|---|
| LightGBM por defecto (5-fold CV) | 0.871 | 0.953 |
| LightGBM ajustado (5-fold CV, umbral 0.5) | 0.886 | 0.962 |
| LightGBM ajustado + umbral 0.41 (out-of-fold) | 0.889 | 0.962 |
| Group K-Fold por reservas duplicadas (escenario exigente) | 0.854 | 0.943 |
| **Test (20%, evaluado una sola vez)** | **0.892** | **0.963** |
| Líder del leaderboard al 11/11 | 0.828 | 0.915 |

Promediar o hacer stacking con XGBoost y CatBoost suma ≈ 0.001–0.002, y por eso se entrega un solo modelo. Para la competencia, el modelo se reentrena con las 47,842 reservas.

## Predictor

```python
from predictor import predictor

predictions = predictor(active_reservations)  # DataFrame (o array) con las 31 columnas de entrada
```

```
python predictor.py reservas.csv   # genera reservas_predictions.csv
```

- Entregables: `predictor.py` + `cancellation_model.joblib` (en la misma carpeta).
- El predictor solo necesita `pandas`, `numpy`, `joblib` y `lightgbm`.
- Las columnas `canceled_job`, `execution_status` y `execution_status_date` se ignoran si vienen en la entrada.

## Cómo ejecutar

```
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

Abrir los notebooks de `notebooks/` con el kernel `.venv` y ejecutarlos en orden. El 1 genera los archivos de `data/` que usan los demás, y el 7 genera `cancellation_model.joblib`. Todo el proyecto usa `random_state = 42`.
