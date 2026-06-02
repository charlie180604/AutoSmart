"""Servicio backend para regresion con red neuronal Keras.

Esta fase replica el flujo de regression_service.py usando TensorFlow/Keras.
Las metricas se calculan en escala real de precio para permitir comparacion
directa contra RandomForestRegressor.
"""

import random

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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
VALIDATION_SPLIT = 0.2
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def _fijar_semillas(random_state: int = RANDOM_STATE) -> None:
    """Reduce variacion entre ejecuciones de numpy, random y TensorFlow."""
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)


def preparar_datos_red_neuronal(autos=None) -> dict:
    """Carga, codifica, separa y escala datos para la red neuronal."""
    if autos is None:
        autos = cargar_dataset()

    columnas_requeridas = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    columnas_faltantes = [
        columna for columna in columnas_requeridas if columna not in autos.columns
    ]
    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas requeridas para red neuronal de regresion: "
            + ", ".join(columnas_faltantes)
        )

    datos = autos[columnas_requeridas].copy()
    datos = datos.dropna()

    for columna in ("anio", "kilometraje", TARGET_COLUMN):
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    datos = datos.dropna()
    datos = datos[datos[TARGET_COLUMN] > 0]

    if datos.empty:
        raise ValueError("No hay registros validos para entrenar red neuronal.")

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

    escalador_x = StandardScaler()
    x_train_escalado = escalador_x.fit_transform(x_train)
    x_test_escalado = escalador_x.transform(x_test)

    escalador_y = StandardScaler()
    y_train_escalado = escalador_y.fit_transform(y_train.to_numpy().reshape(-1, 1))
    y_test_escalado = escalador_y.transform(y_test.to_numpy().reshape(-1, 1))

    return {
        "x": x_codificado,
        "y": y,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "x_train_escalado": x_train_escalado.astype("float32"),
        "x_test_escalado": x_test_escalado.astype("float32"),
        "y_train_escalado": y_train_escalado.astype("float32"),
        "y_test_escalado": y_test_escalado.astype("float32"),
        "escalador_x": escalador_x,
        "escalador_y": escalador_y,
        "variables_utilizadas": list(FEATURE_COLUMNS),
        "variables_codificadas": list(x_codificado.columns),
        "registros_utilizados": int(len(x_codificado)),
    }


def crear_modelo_red_neuronal(input_dim: int) -> Sequential:
    """Crea la arquitectura Keras solicitada para regresion."""
    _fijar_semillas()
    modelo = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    modelo.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=MeanSquaredError(),
    )
    return modelo


def entrenar_modelo_red_neuronal(
    modelo: Sequential,
    x_train,
    y_train,
) -> tf.keras.callbacks.History:
    """Entrena la red con validacion interna y EarlyStopping."""
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )
    return modelo.fit(
        x_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=0,
    )


def calcular_metricas_red_neuronal(y_test, predicciones) -> dict:
    """Calcula MAE, RMSE y R2 con precios en escala real."""
    mse = mean_squared_error(y_test, predicciones)
    return {
        "mae": round(float(mean_absolute_error(y_test, predicciones)), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2": round(float(r2_score(y_test, predicciones)), 4),
    }


def _construir_reales_vs_predichos(y_test, predicciones) -> list[dict]:
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


def generar_chart_data_red_neuronal(
    metricas: dict,
    y_test,
    predicciones,
) -> dict:
    """Prepara datos serializables para graficas futuras."""
    return {
        "metricas": {
            "mae": metricas["mae"],
            "rmse": metricas["rmse"],
            "r2": metricas["r2"],
        },
        "reales_vs_predichos": _construir_reales_vs_predichos(y_test, predicciones),
        "errores": _construir_errores(y_test, predicciones),
    }


def generar_conclusiones_red_neuronal(metricas: dict) -> list[str]:
    """Genera conclusiones automaticas a partir de metricas reales."""
    mae = metricas["mae"]
    rmse = metricas["rmse"]
    r2 = metricas["r2"]

    conclusiones = [
        "La red neuronal estima el precio del vehiculo a partir de ano, kilometraje "
        "y variables categoricas codificadas del dataset.",
        f"El error absoluto medio (MAE) fue de {mae:,.2f}.",
        f"La raiz del error cuadratico medio (RMSE) fue de {rmse:,.2f}.",
        f"El coeficiente de determinacion (R2) fue de {r2:.4f}.",
    ]

    if r2 >= 0.8:
        conclusiones.append(
            "La red neuronal explica una proporcion alta de la variabilidad del precio."
        )
    elif r2 >= 0.5:
        conclusiones.append(
            "La red neuronal captura parte relevante de la variabilidad del precio."
        )
    else:
        conclusiones.append(
            "La red neuronal tiene capacidad limitada para explicar la variabilidad "
            "del precio con las variables disponibles."
        )

    if rmse > mae * 1.35:
        conclusiones.append(
            "La diferencia entre RMSE y MAE indica que existen errores altos en "
            "algunos vehiculos del conjunto de prueba."
        )
    else:
        conclusiones.append(
            "La relacion entre RMSE y MAE sugiere errores relativamente homogeneos."
        )

    return conclusiones


def obtener_contexto_red_neuronal_regresion() -> dict:
    """Ejecuta el flujo completo de regresion con red neuronal Keras."""
    _fijar_semillas()
    datos = preparar_datos_red_neuronal()
    modelo = crear_modelo_red_neuronal(datos["x_train_escalado"].shape[1])
    historial = entrenar_modelo_red_neuronal(
        modelo,
        datos["x_train_escalado"],
        datos["y_train_escalado"],
    )
    predicciones_escaladas = modelo.predict(datos["x_test_escalado"], verbose=0)
    predicciones = datos["escalador_y"].inverse_transform(predicciones_escaladas)
    predicciones = predicciones.ravel()
    metricas = calcular_metricas_red_neuronal(datos["y_test"], predicciones)
    conclusiones = generar_conclusiones_red_neuronal(metricas)
    chart_data = generar_chart_data_red_neuronal(
        metricas,
        datos["y_test"],
        predicciones,
    )

    return {
        "registros_utilizados": datos["registros_utilizados"],
        "variables_utilizadas": datos["variables_utilizadas"],
        "variables_codificadas": datos["variables_codificadas"],
        "variable_objetivo": TARGET_COLUMN,
        "modelo": {
            "nombre": "Red Neuronal Keras",
            "capas": [
                {"tipo": "Dense", "unidades": 64, "activacion": "relu"},
                {"tipo": "Dense", "unidades": 32, "activacion": "relu"},
                {"tipo": "Dense", "unidades": 1, "activacion": "linear"},
            ],
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "loss": "MeanSquaredError",
            "epochs_configurados": EPOCHS,
            "epochs_entrenados": int(len(historial.history.get("loss", []))),
            "batch_size": BATCH_SIZE,
            "validation_split": VALIDATION_SPLIT,
            "early_stopping": {
                "monitor": "val_loss",
                "patience": 20,
                "restore_best_weights": True,
            },
            "random_state": RANDOM_STATE,
        },
        "metricas": metricas,
        "mae": metricas["mae"],
        "rmse": metricas["rmse"],
        "r2": metricas["r2"],
        "historial_entrenamiento": {
            "loss": [round(float(valor), 6) for valor in historial.history.get("loss", [])],
            "val_loss": [
                round(float(valor), 6)
                for valor in historial.history.get("val_loss", [])
            ],
        },
        "conclusiones": conclusiones,
        "chart_data": chart_data,
    }
