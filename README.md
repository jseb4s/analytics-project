# Bank Churn Prediction - Machine Learning Project

Proyecto completo de Machine Learning para predicción de churn bancario, implementando un pipeline desde preprocesamiento hasta evaluación de múltiples modelos.

## Descripción del Proyecto

Este proyecto implementa un sistema completo de predicción de abandono de clientes (churn) en el sector bancario, utilizando técnicas de Machine Learning y siguiendo las mejores prácticas de la industria.

**Dataset:** Bank Customer Churn Dataset (10,127 registros, 21 features)

**Objetivo:** Predecir qué clientes están en riesgo de abandonar el banco

**Modelos implementados:**
- SVM (Baseline y Optimizado con RandomizedSearchCV)
- Random Forest
- LightGBM
- Gradient Boosting
- Neural Network (TensorFlow/Keras)

---

##  Estructura del Proyecto

```
analytics-project/
├── data/
│   ├── raw/                          # Datos originales
│   │   └── bank_churn.xlsx
│   └── processed/                    # Datos preprocesados
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── trained_models/                   # Modelos entrenados (.pkl)
│   ├── preprocessing/
│   │   └── scaler.pkl
│   ├── svm/
│   │   ├── svm_baseline.pkl
│   │   └── svm_optimized.pkl
│   ├── random_forest/
│   │   └── rf_model.pkl
│   ├── lightgbm/
│   │   └── lgbm_model.pkl
│   ├── gradient_boosting/
│   │   └── gb_model.pkl
│   ├── neural_network/
│   │   └── nn_model.pkl
│   └── model_comparison.csv          # Comparación de todos los modelos
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── src/                              # Código fuente
│   ├── __init__.py
│   │
│   ├── utils/                        # Utilidades
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Carga de datos
│   │   └── logger_config.py         # Configuración de logs
│   │
│   ├── preprocessing/                # Preprocesamiento
│   │   ├── __init__.py
│   │   ├── data_cleaning.py         # Limpieza de datos
│   │   └── pipeline.py              # Pipeline completo
│   │
│   └── models/                       # Modelos ML
│       ├── __init__.py
│       ├── base_model.py            # Clase base
│       ├── svm_model.py             # SVM
│       ├── random_forest_model.py   # Random Forest
│       ├── ligthgbm_model.py        # LightGBM
│       ├── gradient_boosting_model.py  # Gradient Boosting
│       ├── neural_network_model.py  # Red Neuronal
│       └── train.py                 # Script de entrenamiento
│
├── reports/                          # Reportes y visualizaciones
│   ├── eda/
│   │   └── eda_report.html
│   └── figures/                      # Gráficos generados
│       ├── svm/
│       ├── random_forest/
│       ├── lightgbm/
│       ├── gradient_boosting/
│       └── neural_network/
│
├── logs/                             # Logs de ejecución
│   └── main_pipeline.log
│
├── main.py                           # 🚀 Script principal
├── pyproject.toml                    # Configuración del proyecto
├── .gitignore
└── README.md
```

---

##  Instalación y Configuración

### Requisitos Previos

- Python 3.12+
- `uv` (gestor de paquetes) o `pip`

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd analytics-project
```

### 2. Crear estructura de carpetas

```bash
mkdir -p data/{raw,processed}
mkdir -p trained_models/{preprocessing,svm,random_forest,lightgbm,gradient_boosting,neural_network}
mkdir -p reports/{eda,figures/{svm,random_forest,lightgbm,gradient_boosting,neural_network}}
mkdir -p logs
```

### 3. Instalar dependencias

```bash
# Con uv (recomendado)
uv add pandas numpy scikit-learn matplotlib seaborn
uv add ydata-profiling[notebook] nbformat
uv add lightgbm
uv add tensorflow scikeras
uv add joblib openpyxl setuptools scipy

# O con requirements.txt
uv pip install -r requirements.txt
```

### 4. Instalar el proyecto en modo editable

```bash
uv pip install -e .
```

### 5. Colocar los datos

```bash
# Mover el archivo de datos a la carpeta correcta
mv bank_churn.xlsx data/raw/
```

---

##  Uso

### Opción 1: Pipeline Completo (Recomendado)

Ejecuta todo el proceso de una sola vez:

```bash
python main.py
```

Esto ejecutará:
1.  Preprocesamiento de datos
2.  Entrenamiento de 6 modelos
3.  Generación de visualizaciones
4.  Comparación de resultados

**Salida esperada:**
```
================================================================================
 INICIANDO PIPELINE COMPLETO DE MACHINE LEARNING
================================================================================

================================================================================
PREPROCESAMIENTO DE DATOS
================================================================================
>>> Cargando datos raw...
Cargando datos desde: data/raw/bank_churn.xlsx
Datos cargados exitosamente: 10127 filas, 21 columnas
...

================================================================================
ENTRENAMIENTO DE MODELOS
================================================================================
>>> Entrenando modelos SVM...
>>> Entrenando Random Forest...
>>> Entrenando LightGBM...
>>> Entrenando Gradient Boosting...
>>> Entrenando Red Neuronal...

================================================================================
RESUMEN FINAL - COMPARACION DE MODELOS
================================================================================

         Modelo  Accuracy  Precision  Recall  F1-Score  ROC-AUC
SVM Optimizado    0.9234     0.8678  0.8123    0.8392   0.9567
...

 MEJOR MODELO: SVM Optimizado
   ROC-AUC: 0.9567

 Pipeline ejecutado exitosamente
```

### Opción 2: Entrenar Modelos Individualmente

**En notebook o script Python:**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))

from models import train_svm_models

# Entrenar solo SVM
svm_results = train_svm_models(save_models=True, save_figures=True)
print(svm_results['comparison'])
```

**Desde terminal:**

```bash
# SVM
python -m models.svm_model

# Random Forest
python -m models.random_forest_model

# LightGBM
python -m models.ligthgbm_model

# Gradient Boosting
python -m models.gradient_boosting_model

# Neural Network
python -m models.neural_network_model
```

### Opción 3: Usar en Jupyter Notebooks

```python
# Setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))

# Preprocesamiento
from utils.data_loader import load_bank_churn_data
from preprocessing import BankChurnPreprocessor, save_processed_data

df = load_bank_churn_data()

categorical_cols = ['gender', 'education_level', 'marital_status', 
                   'income_category', 'card_category']

preprocessor = BankChurnPreprocessor(categorical_cols=categorical_cols)
X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
save_processed_data(X_train, X_test, y_train, y_test)

# Entrenamiento
from models import train_all_models

results = train_all_models(save_models=True, save_figures=True)
print(results['comparison'])
```

---

##  Modelos Implementados

### 1. SVM (Support Vector Machine)

**Dos versiones:**
- **Baseline**: Parámetros por defecto con `class_weight='balanced'`
- **Optimizado**: Búsqueda de hiperparámetros con RandomizedSearchCV

**Hiperparámetros optimizados:**
- C: [0.01, 100]
- gamma: [0.0001, 1]
- kernel: ['linear', 'rbf', 'poly', 'sigmoid']

### 2. Random Forest

**Configuración:**
- n_estimators: 100
- Procesamiento paralelo (n_jobs=-1)

### 3. LightGBM

**Configuración:**
- n_estimators: 200
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8

### 4. Gradient Boosting

**Configuración:**
- n_estimators: 200
- learning_rate: 0.1
- max_depth: 3
- subsample: 0.8

### 5. Neural Network

**Arquitectura:**
- Capa oculta 1: 128 neuronas + Dropout(0.3)
- Capa oculta 2: 64 neuronas + Dropout(0.2)
- Capa de salida: 1 neurona (sigmoid)
- Optimizador: Adam
- Epochs: 30, Batch size: 128

---

##  Preprocesamiento

### Pasos implementados:

1. **Eliminación de columnas:**
   - `clientnum` (ID sin valor predictivo)
   - `credit_limit` (alta correlación)
   - `total_trans_amt` (alta correlación)
   - `total_amt_chng_q4_q1` (alta correlación)
   - `total_revolving_bal` (alta correlación)

2. **Creación de variable objetivo:**
   - `churn = 1` si `attrition_flag == 'Attrited Customer'`
   - `churn = 0` en caso contrario

3. **One-hot encoding:**
   - Variables categóricas: gender, education_level, marital_status, income_category, card_category
   - `drop_first=True` para evitar multicolinealidad

4. **Train-test split:**
   - 80% entrenamiento, 20% test
   - Estratificado por clase

5. **Estandarización:**
   - StandardScaler en columnas numéricas
   - Fit en train, transform en test

---

##  Métricas de Evaluación

Para cada modelo se calculan:

- **Accuracy**: Precisión general
- **Precision**: Precisión de la clase positiva
- **Recall**: Sensibilidad (tasa de verdaderos positivos)
- **F1-Score**: Media armónica de precision y recall
- **ROC-AUC**: Área bajo la curva ROC

### Visualizaciones generadas:

-  Matriz de confusión con porcentajes
-  Curva ROC con AUC
-  Historial de entrenamiento (solo Neural Network)

---

##  Resultados

Después de ejecutar `main.py`, encontrarás:

### Archivos generados:

```
trained_models/
├── model_comparison.csv          # Comparación de todos los modelos
├── preprocessing/scaler.pkl
├── svm/
│   ├── svm_baseline.pkl
│   └── svm_optimized.pkl
├── random_forest/rf_model.pkl
├── lightgbm/lgbm_model.pkl
├── gradient_boosting/gb_model.pkl
└── neural_network/nn_model.pkl

reports/figures/
├── svm/
│   ├── svm_baseline_results.png
│   └── svm_optimized_results.png
├── random_forest/rf_results.png
├── lightgbm/lgbm_results.png
├── gradient_boosting/gb_results.png
└── neural_network/
    ├── nn_results.png
    └── nn_training_history.png
```

### Ver comparación final:

```bash
cat trained_models/model_comparison.csv
```

O en Python:

```python
import pandas as pd
comparison = pd.read_csv('trained_models/model_comparison.csv')
print(comparison.sort_values('ROC-AUC', ascending=False))
```

---

##  Configuración Avanzada

### Personalizar preprocesamiento:

```python
from preprocessing import BankChurnPreprocessor

preprocessor = BankChurnPreprocessor(
    columns_to_drop=['clientnum', 'otra_columna'],
    categorical_cols=['gender', 'education_level'],
    numerical_cols=['customer_age', 'dependent_count']
)
```

### Personalizar modelos:

```python
from models import SVMModel

# SVM con parámetros personalizados
svm = SVMModel(C=10.0, kernel='rbf', gamma=0.001)
svm.train(X_train, y_train)
svm.evaluate(X_test, y_test)
```

### Cargar modelos guardados:

```python
from models import SVMModel
from pathlib import Path

svm = SVMModel()
svm.load_model(Path('trained_models/svm/svm_optimized.pkl'))

# Hacer predicciones
predictions = svm.predict(X_test)
probabilities = svm.predict_proba(X_test)
```

---

##  Logs

Los logs se guardan en:
- `logs/main_pipeline.log` - Log del pipeline completo
- `logs/analytics.log` - Log general del proyecto

**Configuración:**
- Logs se sobrescriben en cada ejecución (`mode='w'`)
- Nivel: INFO
- Formato: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

---

##  Troubleshooting

### Error: ModuleNotFoundError: No module named 'utils'

**Solución:**
```bash
uv pip install -e .
```

### Error: No se encontró el archivo bank_churn.xlsx

**Solución:**
```bash
# Verifica que el archivo esté en la ubicación correcta
ls data/raw/bank_churn.xlsx

# Si no está, muévelo
mv bank_churn.xlsx data/raw/
```

### Error: pkg_resources not found

**Solución:**
```bash
uv pip install setuptools
```

### Conflicto de carpetas models/

**Solución:**
Si tienes una carpeta `models/` en la raíz, renómbrala:
```bash
mv models trained_models
```

---

##  Uso Académico

### Notebooks recomendados:

1. **01_EDA.ipynb** - Análisis exploratorio con ydata-profiling
2. **02_preprocessing.ipynb** - Preprocesamiento paso a paso
3. **03_modeling.ipynb** - Entrenamiento y comparación de modelos

### Estructura de informe sugerida:

1. **Introducción**
2. **Análisis Exploratorio** (EDA)
3. **Preprocesamiento**
   - Limpieza de datos
   - Feature engineering
   - Estandarización
4. **Modelado**
   - Descripción de cada modelo
   - Hiperparámetros
5. **Resultados**
   - Comparación de modelos
   - Mejor modelo
   - Interpretación
6. **Conclusiones**

---

##  Tecnologías Utilizadas

- **Python 3.12**
- **pandas** - Manipulación de datos
- **numpy** - Operaciones numéricas
- **scikit-learn** - Modelos de ML y preprocesamiento
- **LightGBM** - Gradient boosting
- **TensorFlow/Keras** - Redes neuronales
- **matplotlib/seaborn** - Visualizaciones
- **ydata-profiling** - EDA automatizado
- **joblib** - Persistencia de modelos

---

##  Autor

Karen Viviana Duque Angarita

---

**Última actualización:** Noviembre 2025
