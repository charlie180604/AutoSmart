# AutoSmart Advisor

AutoSmart Advisor es una aplicación web desarrollada con Flask que combina un sistema de recomendación de vehículos con módulos de análisis de datos y aprendizaje automático. El proyecto utiliza un dataset procesado de automóviles para ofrecer recomendaciones orientadas al usuario, visualizaciones interactivas y modelos supervisados y no supervisados documentados en interfaz.

## Objetivo

El sistema ayuda a analizar vehículos mediante técnicas de Machine Learning y análisis de datos. A partir de preferencias del usuario (presupuesto, uso, combustible, transmisión, entre otras), genera recomendaciones; además expone módulos académicos para explorar el dataset, reducir dimensionalidad, agrupar registros, clasificar transmisión y estimar precio.

## Características principales

- Página de inicio y formulario de recomendación con evaluación de perfil y riesgo financiero básico.
- Motor de recomendación basado en reglas sobre `data/processed/autos_limpios.csv` (compatibilidad por presupuesto, combustible, transmisión y prioridad).
- Análisis exploratorio de datos (EDA) con estadísticas, correlaciones y gráficas Plotly.
- Módulo PCA con varianza explicada y visualización en componentes principales.
- Clustering con K-Means (método del codo, métricas de calidad e interpretación por segmento).
- Clustering con DBSCAN (detección de clusters y ruido, evaluación de `eps`).
- Clasificación supervisada de transmisión (`manual` / `automatica`) con matriz de confusión e interpretaciones automáticas.
- Regresión supervisada de precio con métricas MAE, RMSE y R², gráficas de predicción y interpretaciones automáticas.
- Centro de Visualización ML: hub de navegación hacia todos los algoritmos implementados.
- Sección Extras con acceso al hub y apartados complementarios planificados.
- Página Acerca del proyecto.

## Roadmap

### Versión actual

- Análisis exploratorio de datos
- PCA
- K-Means
- DBSCAN
- Clasificación supervisada
- Regresión supervisada
- Centro de Visualización ML

### Próximas mejoras

- Red Neuronal Artificial para Regresión
- Comparación avanzada de modelos
- Recomendaciones inteligentes basadas en perfiles de usuario

## Tecnologías utilizadas

### Frontend

- HTML5
- CSS3 (estilos embebidos por plantilla)
- Bootstrap 5.3.3
- Bootstrap Icons 1.11.3
- Google Fonts (Inter)

### Backend

- Python 3
- Flask

### Machine Learning y datos

- pandas
- numpy
- scikit-learn (`PCA`, `KMeans`, `DBSCAN`, `RandomForestClassifier`, `RandomForestRegressor`, `StandardScaler`, métricas y `train_test_split`)
- joblib (dependencia declarada en `requirements.txt`)

### Visualización

- Plotly 2.35.2 (CDN en plantillas de análisis y algoritmos)
- Matplotlib (`ml/matplotlib_analysis.py`, generación de gráficas estáticas; no integrado en rutas Flask)

### Base de datos

No se utiliza base de datos relacional. Los datos se almacenan y consumen desde archivos CSV en `data/processed/` y fuentes en `data/raw/`.

## Estructura del proyecto

```
AutoSmartAdvisor/
├── app.py                          # Aplicación Flask y rutas
├── requirements.txt                # Dependencias Python
├── data/
│   ├── raw/                        # Datasets originales
│   └── processed/
│       └── autos_limpios.csv       # Dataset principal del sistema
├── ml/
│   ├── preprocessing.py             # Limpieza y exportación del CSV procesado
│   ├── exploratory_analysis.py      # Funciones base de EDA
│   ├── analysis_service.py          # Contexto para /analisis
│   ├── pca_service.py               # PCA
│   ├── kmeans_service.py            # K-Means
│   ├── dbscan_service.py            # DBSCAN
│   ├── classification_service.py    # Clasificación
│   ├── regression_service.py        # Regresión
│   ├── regression_interpretation.py # Textos interpretativos de regresión
│   ├── predictor.py                 # Recomendación por reglas
│   ├── matplotlib_analysis.py       # Gráficas EDA con Matplotlib
│   └── train_models.py              # Placeholder para entrenamientos futuros
└── templates/
    ├── index.html
    ├── formulario.html
    ├── resultados.html
    ├── analisis.html
    ├── pca.html
    ├── kmeans.html
    ├── dbscan.html
    ├── clasificacion.html
    ├── regresion.html
    ├── centro_visualizacion_ml.html
    ├── extras.html
    └── acerca.html
```

## Dataset utilizado

| Aspecto | Detalle |
|---------|---------|
| Archivo principal | `data/processed/autos_limpios.csv` |
| Registros | 3 573 vehículos |
| Columnas | 12: `nombre`, `anio`, `precio`, `kilometraje`, `combustible`, `tipo_vendedor`, `transmision`, `propietarios`, `tipo`, `rendimiento`, `pasajeros`, `etiqueta` |
| Origen | Procesado desde archivos en `data/raw/` mediante `ml/preprocessing.py` |

### Variables principales

- **Identificación y precio:** `nombre`, `precio`, `anio`, `kilometraje`
- **Características del vehículo:** `combustible`, `transmision`, `tipo`, `rendimiento`, `pasajeros`
- **Contexto de venta:** `tipo_vendedor`, `propietarios`
- **Etiqueta de recomendación:** `etiqueta`

### Variables objetivo en modelos supervisados

| Modelo | Variable objetivo | Variables de entrada (según servicio) |
|--------|-------------------|--------------------------------------|
| Clasificación | `transmision` (manual / automatica) | `precio`, `anio`, `kilometraje`, `combustible`, `tipo_vendedor`, `propietarios` |
| Regresión | `precio` | `anio`, `kilometraje`, `combustible`, `tipo_vendedor`, `propietarios` |

Ambos modelos supervisados usan partición entrenamiento/prueba 80/20 con `random_state=42` y codificación de categóricas con `pd.get_dummies()`.

## Algoritmos implementados

| Algoritmo | Tipo | Propósito |
|-----------|------|-----------|
| PCA | Reducción de dimensionalidad | Reducir y analizar variables numéricas (`precio`, `anio`, `kilometraje`, `rendimiento`, `pasajeros`) antes de visualizaciones y clustering |
| K-Means | Aprendizaje no supervisado | Agrupar vehículos en clusters sobre componentes PCA (por defecto 3 clusters) |
| DBSCAN | Aprendizaje no supervisado | Detectar agrupaciones por densidad y puntos de ruido sobre PCA (por defecto `eps=0.8`, `min_samples=8`) |
| Random Forest (clasificación) | Aprendizaje supervisado | Predecir `transmision` del vehículo |
| Random Forest (regresión) | Aprendizaje supervisado | Estimar `precio` del vehículo |
| Red Neuronal Artificial | Aprendizaje profundo | En desarrollo (`train_models.py` y tarjeta del Centro de Visualización ML) |

## Módulos del sistema

### Recomendación (`/formulario`, `/resultados`)

- Formulario de captura de datos del usuario.
- `ml/predictor.py` genera hasta 3 recomendaciones desde el CSV procesado.
- Evaluación de perfil de uso y nivel de riesgo financiero en `app.py`.
- No utiliza modelos entrenados de scikit-learn; aplica reglas de compatibilidad documentadas en el código.

### Análisis (`/analisis`)

- EDA servido por `ml/analysis_service.py` y `ml/exploratory_analysis.py`.
- Incluye estadísticas descriptivas, correlaciones, detección de valores atípicos y gráficas Plotly.

### Extras (`/extras`)

- Recursos complementarios del proyecto.
- Acceso al Centro de Visualización ML.
- Tarjetas con enlaces a `/comparacion`, `/metricas`, `/dataset` y `/pruebas` aún sin ruta Flask registrada.

### Centro de Visualización ML (`/centro-visualizacion`)

- Hub estático de navegación hacia PCA, K-Means, DBSCAN, Clasificación y Regresión.
- Incluye diagrama de arquitectura general y módulo de Red Neuronal marcado como próximo.

### Predictor (`ml/predictor.py`)

- Función principal: `generar_recomendacion(datos_usuario, cantidad=3)`.
- Garantiza la existencia del dataset procesado; puede invocar `preparar_autos()` si el CSV no existe.

## Métricas implementadas

### Clasificación

| Métrica | Descripción breve |
|---------|-------------------|
| **Accuracy** | Proporción de predicciones correctas sobre el total del conjunto de prueba. |
| **Precision** | De las predicciones positivas por clase, fracción que fue correcta (promedio ponderado). |
| **Recall** | De los casos reales de cada clase, fracción que el modelo identificó correctamente. |
| **F1 Score** | Media armónica entre precision y recall; útil con posible desbalance de clases. |
| **Matriz de confusión** | Tabla que contrasta etiquetas reales y predichas para `manual` y `automatica`. |

### Regresión

| Métrica | Descripción breve |
|---------|-------------------|
| **MAE** | Error absoluto medio: desviación promedio en unidades de precio entre valor real y predicho. |
| **RMSE** | Raíz del error cuadrático medio: penaliza más los errores grandes que el MAE. |
| **R²** | Coeficiente de determinación: proporción de variabilidad del precio explicada por el modelo en prueba. |

## Capturas sugeridas

Inserte capturas de pantalla en una carpeta como `docs/images/` y referéncielas en este README:

```markdown
![Inicio](docs/images/inicio.png)
```

| Vista | Ruta sugerida | URL local |
|-------|---------------|-----------|
| Inicio | `docs/images/inicio.png` | `/` |
| Recomendación (formulario / resultados) | `docs/images/recomendacion.png` | `/formulario`, `/resultados` |
| PCA | `docs/images/pca.png` | `/pca` |
| K-Means | `docs/images/kmeans.png` | `/kmeans` |
| DBSCAN | `docs/images/dbscan.png` | `/dbscan` |
| Clasificación | `docs/images/clasificacion.png` | `/clasificacion` |
| Regresión | `docs/images/regresion.png` | `/regresion` |
| Centro de Visualización ML | `docs/images/centro-visualizacion.png` | `/centro-visualizacion` |

## Instalación

### Requisitos previos

- Python 3.10 o superior recomendado
- pip

### Pasos

1. **Clonar el repositorio**

```bash
git clone <url-del-repositorio>
cd AutoSmartAdvisor/AutoSmartAdvisor
```

2. **Crear entorno virtual**

```bash
python -m venv .venv
```

En Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

3. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

4. **Ejecutar la aplicación**

Desde la carpeta que contiene `app.py`:

```bash
python app.py
```

5. **Abrir en el navegador**

```
http://127.0.0.1:5000
```

La aplicación se inicia con `debug=True` según la configuración actual de `app.py`.

### Dataset procesado

Si `data/processed/autos_limpios.csv` no existe, el recomendador puede generarlo ejecutando el preprocesamiento definido en `ml/preprocessing.py` (requiere un archivo fuente en `data/raw/`).

## Rutas disponibles

| Ruta | Método | Descripción |
|------|--------|-------------|
| `/` | GET | Página de inicio |
| `/formulario` | GET | Formulario de recomendación |
| `/resultados` | POST | Resultados de recomendación (GET redirige a formulario) |
| `/analisis` | GET | Análisis exploratorio de datos |
| `/pca` | GET | Análisis de componentes principales |
| `/kmeans` | GET | Clustering K-Means |
| `/dbscan` | GET | Clustering DBSCAN |
| `/clasificacion` | GET | Clasificación de transmisión |
| `/regresion` | GET | Regresión de precio |
| `/centro-visualizacion` | GET | Hub de algoritmos ML |
| `/extras` | GET | Recursos complementarios |
| `/acerca` | GET | Información del proyecto |

## Estado actual del proyecto

| Módulo | Estado |
|--------|--------|
| Inicio y navegación principal | Completado |
| Recomendación (predictor por reglas) | Completado |
| Análisis exploratorio (`/analisis`) | Completado |
| PCA (`/pca`) | Completado |
| K-Means (`/kmeans`) | Completado |
| DBSCAN (`/dbscan`) | Completado |
| Clasificación (`/clasificacion`) | Completado |
| Regresión (`/regresion`) | Completado |
| Centro de Visualización ML (`/centro-visualizacion`) | Completado |
| Integración Extras → Hub ML | Completado |
| Red Neuronal Artificial | En desarrollo |
| Gráficas Matplotlib en web | Pendiente (módulo `matplotlib_analysis.py` existente) |
| Extras: comparación (`/comparacion`) | Pendiente |
| Extras: métricas (`/metricas`) | Pendiente |
| Extras: dataset (`/dataset`) | Pendiente |
| Extras: pruebas (`/pruebas`) | Pendiente |
| Entrenamiento centralizado (`train_models.py`) | Pendiente |

Actualmente el proyecto implementa múltiples técnicas de aprendizaje automático para el análisis de vehículos y la estimación de precios.

La implementación de una Red Neuronal Artificial para regresión se encuentra planificada como siguiente fase de desarrollo y será incorporada en futuras versiones del sistema.

## Relación con aprendizaje automático

| Componente | Rol en el sistema |
|------------|-------------------|
| **Dataset procesado** | Fuente única para EDA, modelos y recomendación |
| **EDA** | Comprensión de distribuciones, correlaciones y valores atípicos antes del modelado |
| **PCA** | Reduce dimensionalidad y alimenta visualizaciones; precede K-Means y DBSCAN |
| **K-Means** | Segmenta vehículos similares para análisis de patrones |
| **DBSCAN** | Identifica clusters de densidad variable y registros atípicos (ruido) |
| **Clasificación** | Modelo supervisado para predecir tipo de transmisión |
| **Regresión** | Modelo supervisado para estimar precio con evaluación cuantitativa |
| **Centro de Visualización ML** | Organiza el acceso académico a cada módulo sin reentrenar modelos |
| **Recomendación** | Capa de decisión para el usuario final; actualmente basada en reglas, no en los modelos ML de la interfaz |

## Trabajo académico

El proyecto integra contenidos típicos de una asignatura de Machine Learning o ciencia de datos:

| Tema académico | Implementación en AutoSmart Advisor |
|----------------|-------------------------------------|
| Análisis exploratorio | `/analisis`, `exploratory_analysis.py`, `analysis_service.py` |
| Reducción de dimensionalidad | `/pca`, `pca_service.py` |
| Clustering particional | `/kmeans`, `kmeans_service.py` |
| Clustering por densidad | `/dbscan`, `dbscan_service.py` |
| Clasificación supervisada | `/clasificacion`, `classification_service.py` |
| Regresión supervisada | `/regresion`, `regression_service.py` |
| Evaluación de modelos | Métricas, matrices de confusión, gráficas reales vs predichos, interpretaciones automáticas |
| Interfaz web de resultados | Plantillas Flask con Bootstrap y Plotly; hub en `/centro-visualizacion` |

## Autor

**Nombre(s):** [Realizado por un equipo]

**Institución:** [Instituto Tecnologico de Minatitlan]

**Año:** 2026
