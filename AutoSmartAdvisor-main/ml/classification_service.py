"""Servicio backend para clasificacion supervisada de transmision.

Esta fase entrena un RandomForestClassifier para predecir la variable
objetivo ``transmision`` sin modificar el recomendador ni la interfaz.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from ml.exploratory_analysis import cargar_dataset


FEATURE_COLUMNS = (
    "precio",
    "anio",
    "kilometraje",
    "combustible",
    "tipo_vendedor",
    "propietarios",
)
TARGET_COLUMN = "transmision"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def _distribution_from_series(serie: pd.Series, clases_ordenadas: list[str]) -> dict:
    """Convierte una serie de etiquetas en conteos y porcentajes ordenados."""
    conteos = serie.value_counts().to_dict()
    total = int(len(serie))
    return {
        "clases": clases_ordenadas,
        "conteos": [int(conteos.get(clase, 0)) for clase in clases_ordenadas],
        "porcentajes": [
            round((int(conteos.get(clase, 0)) / total) * 100, 2) if total else 0.0
            for clase in clases_ordenadas
        ],
        "total": total,
    }


def _extraer_importancia_variables(modelo: RandomForestClassifier, columnas: list[str]) -> dict:
    """Empaqueta feature_importances_ para consumo visual."""
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


def _construir_predicciones_detalladas(
    x_raw_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series | list,
    limite: int = 250,
) -> dict:
    """Devuelve ejemplos de predicciones correctas e incorrectas para UI."""
    y_pred_serie = pd.Series(y_pred, index=y_test.index, name="y_pred")
    comparacion = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred_serie,
        }
    )
    comparacion["correcta"] = comparacion["y_true"] == comparacion["y_pred"]
    comparacion = comparacion.join(x_raw_test, how="left")

    def _a_registro(fila) -> dict:
        registro = {
            "index": int(fila.name) if str(fila.name).isdigit() else str(fila.name),
            "y_true": str(fila["y_true"]),
            "y_pred": str(fila["y_pred"]),
            "correcta": bool(fila["correcta"]),
        }
        for columna in FEATURE_COLUMNS:
            if columna in fila:
                valor = fila[columna]
                registro[columna] = None if pd.isna(valor) else valor
        return registro

    correctas = comparacion[comparacion["correcta"]].head(limite)
    incorrectas = comparacion[~comparacion["correcta"]].head(limite)

    return {
        "resumen": {
            "total": int(len(comparacion)),
            "correctas": int(comparacion["correcta"].sum()),
            "incorrectas": int((~comparacion["correcta"]).sum()),
            "limite_mostrado": int(limite),
        },
        "correctas": [_a_registro(fila) for _, fila in correctas.iterrows()],
        "incorrectas": [_a_registro(fila) for _, fila in incorrectas.iterrows()],
    }


def preparar_datos_clasificacion(autos=None) -> dict:
    """Selecciona variables, limpia nulos, codifica categoricas y separa X/y."""
    if autos is None:
        autos = cargar_dataset()

    columnas_requeridas = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    columnas_faltantes = [
        columna for columna in columnas_requeridas if columna not in autos.columns
    ]
    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas requeridas para clasificacion: "
            + ", ".join(columnas_faltantes)
        )

    datos = autos[columnas_requeridas].copy()
    datos = datos.dropna()
    datos = datos[datos[TARGET_COLUMN].isin(["manual", "automatica"])]

    if datos.empty:
        raise ValueError("No hay registros validos para entrenar clasificacion.")

    x = datos[list(FEATURE_COLUMNS)].copy()
    y = datos[TARGET_COLUMN].copy()

    columnas_numericas = ["precio", "anio", "kilometraje"]
    for columna in columnas_numericas:
        x[columna] = pd.to_numeric(x[columna], errors="coerce")

    x = x.dropna()
    y = y.loc[x.index]

    x_raw = x.copy()
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
        stratify=y,
    )

    return {
        "x": x_codificado,
        "y": y,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "x_raw": x_raw,
        "x_train_raw": x_raw.loc[x_train.index],
        "x_test_raw": x_raw.loc[x_test.index],
        "variables_utilizadas": list(FEATURE_COLUMNS),
        "variables_codificadas": list(x_codificado.columns),
        "clases_detectadas": sorted(y.unique().tolist()),
        "registros_utilizados": int(len(x_codificado)),
    }


def entrenar_modelo_clasificacion(x_train, y_train) -> RandomForestClassifier:
    """Entrena RandomForestClassifier con parametros razonables para practica."""
    modelo = RandomForestClassifier(
        n_estimators=120,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    modelo.fit(x_train, y_train)
    return modelo


def calcular_metricas_clasificacion(y_test, y_pred) -> dict:
    """Calcula metricas principales de clasificacion."""
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(
            float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            4,
        ),
        "recall": round(
            float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            4,
        ),
        "f1_score": round(
            float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            4,
        ),
    }


def obtener_matriz_confusion(y_test, y_pred, clases: list[str]) -> dict:
    """Genera matriz de confusion en estructura reutilizable."""
    matriz = confusion_matrix(y_test, y_pred, labels=clases)
    return {
        "clases": clases,
        "matriz": matriz.tolist(),
    }


def _descriptor_transmision(clase: str, plural: bool = False) -> str:
    """Devuelve un descriptor legible para la clase de transmision."""
    descriptores = {
        "manual": ("manual", "manuales"),
        "automatica": ("automatica", "automaticas"),
    }
    singular, plural_form = descriptores.get(clase, (clase, clase))
    return plural_form if plural else singular


def generar_interpretacion_matriz_confusion(
    matriz_confusion: dict,
    distribucion_clases: dict | None = None,
) -> list[str]:
    """Genera conclusiones automaticas a partir de la matriz de confusion real."""
    clases = matriz_confusion.get("clases", [])
    matriz = matriz_confusion.get("matriz", [])
    if not clases or not matriz:
        return []

    interpretaciones = []

    for indice, clase in enumerate(clases):
        correctos = int(matriz[indice][indice])
        if correctos > 0:
            interpretaciones.append(
                "El modelo clasifico correctamente "
                f"{correctos} vehiculos con transmision {_descriptor_transmision(clase)}."
            )

    errores = []
    for indice_real, clase_real in enumerate(clases):
        for indice_pred, clase_pred in enumerate(clases):
            if indice_real == indice_pred:
                continue
            cantidad = int(matriz[indice_real][indice_pred])
            if cantidad <= 0:
                continue
            interpretaciones.append(
                f"Se detectaron {cantidad} vehiculos "
                f"{_descriptor_transmision(clase_real, plural=True)} clasificados como "
                f"{_descriptor_transmision(clase_pred)}."
            )
            errores.append(
                {
                    "cantidad": cantidad,
                    "clase_real": clase_real,
                    "clase_pred": clase_pred,
                }
            )

    if errores:
        total_errores = sum(error["cantidad"] for error in errores)
        error_principal = max(errores, key=lambda error: error["cantidad"])
        if total_errores > 0:
            if len(errores) == 1 or error_principal["cantidad"] / total_errores >= 0.5:
                interpretaciones.append(
                    "La mayor parte de los errores ocurre al confundir transmisiones "
                    f"{_descriptor_transmision(error_principal['clase_real'], plural=True)} "
                    f"con {_descriptor_transmision(error_principal['clase_pred'])}."
                )

    if distribucion_clases:
        porcentajes = distribucion_clases.get("porcentajes", [])
        if len(porcentajes) >= 2:
            maximo = max(porcentajes)
            minimo = min(porcentajes)
            if maximo >= 70 or (minimo > 0 and maximo / minimo >= 3):
                interpretaciones.append(
                    "El conjunto de datos presenta un desbalance importante entre ambas clases."
                )

    return interpretaciones


def generar_conclusiones_clasificacion(metricas: dict, clases: list[str]) -> list[str]:
    """Genera conclusiones automaticas basadas en metricas reales."""
    accuracy = metricas["accuracy"]
    f1 = metricas["f1_score"]

    conclusiones = [
        "El modelo de clasificacion intenta predecir la transmision del vehiculo "
        f"entre las clases: {', '.join(clases)}.",
        f"El accuracy obtenido fue de {accuracy * 100:.2f}%.",
        f"El F1 Score ponderado fue de {f1 * 100:.2f}%, combinando precision y recall.",
    ]

    if accuracy >= 0.85:
        conclusiones.append(
            "El desempeno general es alto para una primera practica supervisada."
        )
    elif accuracy >= 0.7:
        conclusiones.append(
            "El desempeno general es aceptable, aunque puede mejorar con mas variables."
        )
    else:
        conclusiones.append(
            "El desempeno general es limitado; conviene revisar variables y balance de clases."
        )

    return conclusiones


def obtener_contexto_clasificacion() -> dict:
    """Funcion principal para ejecutar el flujo completo de clasificacion."""
    datos = preparar_datos_clasificacion()
    modelo = entrenar_modelo_clasificacion(datos["x_train"], datos["y_train"])
    predicciones = modelo.predict(datos["x_test"])
    metricas = calcular_metricas_clasificacion(datos["y_test"], predicciones)
    matriz_confusion = obtener_matriz_confusion(
        datos["y_test"],
        predicciones,
        datos["clases_detectadas"],
    )
    conclusiones = generar_conclusiones_clasificacion(
        metricas,
        datos["clases_detectadas"],
    )

    distribucion_total = _distribution_from_series(datos["y"], datos["clases_detectadas"])
    distribucion_train = _distribution_from_series(
        datos["y_train"], datos["clases_detectadas"]
    )
    distribucion_test = _distribution_from_series(
        datos["y_test"], datos["clases_detectadas"]
    )
    interpretacion_matriz_confusion = generar_interpretacion_matriz_confusion(
        matriz_confusion,
        distribucion_total,
    )
    importancia_variables = _extraer_importancia_variables(
        modelo,
        datos["variables_codificadas"],
    )
    predicciones_detalle = _construir_predicciones_detalladas(
        datos["x_test_raw"],
        datos["y_test"],
        predicciones,
    )

    chart_data = {
        "confusion_matrix": matriz_confusion,
        "distribucion_clases": {
            "total": distribucion_total,
            "train": distribucion_train,
            "test": distribucion_test,
        },
        "importancia_variables": importancia_variables,
        "metricas": {
            "accuracy": metricas["accuracy"],
            "precision": metricas["precision"],
            "recall": metricas["recall"],
            "f1_score": metricas["f1_score"],
        },
        "predicciones": predicciones_detalle,
    }

    return {
        "registros_utilizados": datos["registros_utilizados"],
        "variables_utilizadas": datos["variables_utilizadas"],
        "variables_codificadas": datos["variables_codificadas"],
        "clases_detectadas": datos["clases_detectadas"],
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "modelo": {
            "nombre": "RandomForestClassifier",
            "n_estimators": modelo.n_estimators,
            "min_samples_split": modelo.min_samples_split,
            "min_samples_leaf": modelo.min_samples_leaf,
            "class_weight": modelo.class_weight,
        },
        "accuracy": metricas["accuracy"],
        "precision": metricas["precision"],
        "recall": metricas["recall"],
        "f1_score": metricas["f1_score"],
        "matriz_confusion": matriz_confusion,
        "distribucion_clases": {
            "total": distribucion_total,
            "train": distribucion_train,
            "test": distribucion_test,
        },
        "importancia_variables": importancia_variables,
        "predicciones": predicciones_detalle,
        "interpretacion_matriz_confusion": interpretacion_matriz_confusion,
        "conclusiones": conclusiones,
        "chart_data": chart_data,
    }
