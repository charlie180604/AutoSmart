"""Servicio backend para regresion supervisada de precio.

Esta fase entrena un RandomForestRegressor para predecir el precio de vehiculos
sin modificar el recomendador ni la interfaz web.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ml.exploratory_analysis import cargar_dataset


FEATURE_COLUMNS = (
    "anio",
    "kilometraje",
    "combustible",
    "tipo_vendedor",
    "propietarios",
)
TARGET_COLUMN = "precio"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def preparar_datos_regresion(autos=None) -> dict:
    """Carga el dataset, prepara variables y separa entrenamiento/prueba."""
    if autos is None:
        autos = cargar_dataset()

    columnas_requeridas = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    columnas_faltantes = [
        columna for columna in columnas_requeridas if columna not in autos.columns
    ]
    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas requeridas para regresion: " + ", ".join(columnas_faltantes)
        )

    datos = autos[columnas_requeridas].copy()
    datos = datos.dropna()

    for columna in ("anio", "kilometraje", TARGET_COLUMN):
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    datos = datos.dropna()
    datos = datos[datos[TARGET_COLUMN] > 0]

    if datos.empty:
        raise ValueError("No hay registros validos para entrenar regresion.")

    x = datos[list(FEATURE_COLUMNS)].copy()
    y = datos[TARGET_COLUMN].copy()

    x_codificado = pd.get_dummies(
        x,
        columns=["combustible", "tipo_vendedor", "propietarios"],
        drop_first=False,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x_codificado,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    return {
        "x": x_codificado,
        "y": y,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "variables_utilizadas": list(FEATURE_COLUMNS),
        "variables_codificadas": list(x_codificado.columns),
        "registros_utilizados": int(len(x_codificado)),
    }


def entrenar_modelo_regresion(x_train, y_train) -> RandomForestRegressor:
    """Entrena RandomForestRegressor con parametros razonables para practica."""
    modelo = RandomForestRegressor(
        n_estimators=120,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    )
    modelo.fit(x_train, y_train)
    return modelo


def calcular_metricas_regresion(y_test, y_pred) -> dict:
    """Calcula metricas principales de regresion."""
    mse = mean_squared_error(y_test, y_pred)
    return {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
    }


def obtener_importancia_variables(
    modelo: RandomForestRegressor,
    columnas: list[str],
) -> dict:
    """Empaqueta feature_importances_ del modelo entrenado."""
    importancias = [float(valor) for valor in getattr(modelo, "feature_importances_", [])]
    pares = list(zip(columnas, importancias, strict=False))
    pares_ordenados = sorted(pares, key=lambda item: item[1], reverse=True)

    return {
        "variables": [nombre for nombre, _ in pares_ordenados],
        "importancias": [round(valor, 6) for _, valor in pares_ordenados],
        "top": [
            {"variable": nombre, "importancia": round(valor, 6)}
            for nombre, valor in pares_ordenados[:20]
        ],
    }


def generar_conclusiones_regresion(metricas: dict, importancia_variables: dict) -> list[str]:
    """Genera conclusiones automaticas a partir de metricas reales."""
    mae = metricas["mae"]
    rmse = metricas["rmse"]
    r2 = metricas["r2"]

    conclusiones = [
        "El modelo de regresion estima el precio del vehiculo a partir de "
        "ano, kilometraje y variables categoricas del dataset.",
        f"El error absoluto medio (MAE) fue de {mae:,.2f}.",
        f"La raiz del error cuadratico medio (RMSE) fue de {rmse:,.2f}.",
        f"El coeficiente de determinacion (R2) fue de {r2:.4f}.",
    ]

    if r2 >= 0.8:
        conclusiones.append(
            "El modelo explica una proporcion alta de la variabilidad del precio."
        )
    elif r2 >= 0.5:
        conclusiones.append(
            "El modelo captura parte relevante de la variabilidad del precio."
        )
    else:
        conclusiones.append(
            "El modelo tiene capacidad limitada para explicar la variabilidad del precio."
        )

    top = importancia_variables.get("top", [])
    if top:
        variable_principal = top[0]["variable"]
        conclusiones.append(
            "La variable con mayor influencia en las predicciones fue "
            f"{variable_principal}."
        )

    return conclusiones


def _construir_reales_vs_predichos(y_test, predicciones) -> list[dict]:
    """Lista precio real y predicho por cada registro del conjunto de prueba."""
    resultado = []
    for precio_real, precio_predicho in zip(y_test.values, predicciones, strict=False):
        resultado.append(
            {
                "precio_real": round(float(precio_real), 2),
                "precio_predicho": round(float(precio_predicho), 2),
            }
        )
    return resultado


def _construir_errores(y_test, predicciones) -> list[dict]:
    """Lista error (residual), precio real y predicho para analisis posterior."""
    resultado = []
    for precio_real, precio_predicho in zip(y_test.values, predicciones, strict=False):
        real = float(precio_real)
        predicho = float(precio_predicho)
        resultado.append(
            {
                "error": round(real - predicho, 2),
                "precio_real": round(real, 2),
                "precio_predicho": round(predicho, 2),
            }
        )
    return resultado


def generar_chart_data_regresion(
    metricas: dict,
    importancia_variables: dict,
    y_test,
    predicciones,
) -> dict:
    """Prepara estructuras serializables para visualizacion Plotly en fases posteriores."""
    return {
        "metricas": {
            "mae": metricas["mae"],
            "rmse": metricas["rmse"],
            "r2": metricas["r2"],
        },
        "importancia_variables": importancia_variables,
        "reales_vs_predichos": _construir_reales_vs_predichos(y_test, predicciones),
        "errores": _construir_errores(y_test, predicciones),
    }


def obtener_contexto_regresion() -> dict:
    """Funcion principal que ejecuta el flujo completo de regresion."""
    datos = preparar_datos_regresion()
    modelo = entrenar_modelo_regresion(datos["x_train"], datos["y_train"])
    predicciones = modelo.predict(datos["x_test"])
    metricas = calcular_metricas_regresion(datos["y_test"], predicciones)
    importancia_variables = obtener_importancia_variables(
        modelo,
        datos["variables_codificadas"],
    )
    conclusiones = generar_conclusiones_regresion(metricas, importancia_variables)
    chart_data = generar_chart_data_regresion(
        metricas,
        importancia_variables,
        datos["y_test"],
        predicciones,
    )

    return {
        "registros_utilizados": datos["registros_utilizados"],
        "variables_utilizadas": datos["variables_utilizadas"],
        "variable_objetivo": TARGET_COLUMN,
        "modelo": {
            "nombre": "RandomForestRegressor",
            "n_estimators": modelo.n_estimators,
            "min_samples_split": modelo.min_samples_split,
            "min_samples_leaf": modelo.min_samples_leaf,
            "random_state": RANDOM_STATE,
        },
        "metricas": metricas,
        "mae": metricas["mae"],
        "rmse": metricas["rmse"],
        "r2": metricas["r2"],
        "importancia_variables": importancia_variables,
        "conclusiones": conclusiones,
        "chart_data": chart_data,
    }
