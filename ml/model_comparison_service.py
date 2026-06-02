"""Servicio para comparar modelos de regresion ya implementados."""

from ml.neural_regression_service import obtener_contexto_red_neuronal_regresion
from ml.regression_service import obtener_contexto_regresion


MODELO_RANDOM_FOREST = "Random Forest"
MODELO_RED_NEURONAL = "Red Neuronal"


def _formatear_moneda(valor: float) -> str:
    return f"${valor:,.0f}"


def _ganador_error(valor_random_forest: float, valor_red_neuronal: float) -> str:
    if valor_random_forest < valor_red_neuronal:
        return MODELO_RANDOM_FOREST
    if valor_red_neuronal < valor_random_forest:
        return MODELO_RED_NEURONAL
    return "Empate"


def _ganador_r2(valor_random_forest: float, valor_red_neuronal: float) -> str:
    if valor_random_forest > valor_red_neuronal:
        return MODELO_RANDOM_FOREST
    if valor_red_neuronal > valor_random_forest:
        return MODELO_RED_NEURONAL
    return "Empate"


def _comparar_metrica(
    clave: str,
    nombre: str,
    valor_random_forest: float,
    valor_red_neuronal: float,
    mayor_es_mejor: bool,
) -> dict:
    ganador = (
        _ganador_r2(valor_random_forest, valor_red_neuronal)
        if mayor_es_mejor
        else _ganador_error(valor_random_forest, valor_red_neuronal)
    )
    return {
        "clave": clave,
        "nombre": nombre,
        "random_forest": round(float(valor_random_forest), 4),
        "red_neuronal": round(float(valor_red_neuronal), 4),
        "diferencia_absoluta": round(
            abs(float(valor_random_forest) - float(valor_red_neuronal)),
            4,
        ),
        "ganador": ganador,
        "criterio": "Mayor es mejor" if mayor_es_mejor else "Menor es mejor",
    }


def construir_comparacion_metricas(
    metricas_random_forest: dict,
    metricas_red_neuronal: dict,
) -> list[dict]:
    """Construye la tabla comparativa para MAE, RMSE y R2."""
    return [
        _comparar_metrica(
            "mae",
            "MAE",
            metricas_random_forest["mae"],
            metricas_red_neuronal["mae"],
            mayor_es_mejor=False,
        ),
        _comparar_metrica(
            "rmse",
            "RMSE",
            metricas_random_forest["rmse"],
            metricas_red_neuronal["rmse"],
            mayor_es_mejor=False,
        ),
        _comparar_metrica(
            "r2",
            "R2",
            metricas_random_forest["r2"],
            metricas_red_neuronal["r2"],
            mayor_es_mejor=True,
        ),
    ]


def determinar_ganadores(comparacion_metricas: list[dict]) -> dict:
    """Determina ganadores por metrica y ganador global por mayoria."""
    ganadores = {
        item["clave"]: item["ganador"]
        for item in comparacion_metricas
    }
    conteo = {
        MODELO_RANDOM_FOREST: 0,
        MODELO_RED_NEURONAL: 0,
    }
    for ganador in ganadores.values():
        if ganador in conteo:
            conteo[ganador] += 1

    if conteo[MODELO_RANDOM_FOREST] > conteo[MODELO_RED_NEURONAL]:
        ganador_global = MODELO_RANDOM_FOREST
    elif conteo[MODELO_RED_NEURONAL] > conteo[MODELO_RANDOM_FOREST]:
        ganador_global = MODELO_RED_NEURONAL
    else:
        ganador_global = "Empate"

    return {
        "mae": ganadores["mae"],
        "rmse": ganadores["rmse"],
        "r2": ganadores["r2"],
        "global": ganador_global,
        "conteo": conteo,
    }


def generar_chart_data_comparacion(comparacion_metricas: list[dict]) -> dict:
    """Prepara datos serializables para graficas comparativas."""
    return {
        "metricas": [item["nombre"] for item in comparacion_metricas],
        "random_forest": [item["random_forest"] for item in comparacion_metricas],
        "red_neuronal": [item["red_neuronal"] for item in comparacion_metricas],
        "ganadores": [item["ganador"] for item in comparacion_metricas],
    }


def generar_conclusiones_comparacion(
    comparacion_metricas: list[dict],
    ganadores: dict,
) -> list[str]:
    """Genera conclusiones automaticas para usuarios no tecnicos."""
    por_clave = {item["clave"]: item for item in comparacion_metricas}
    mae = por_clave["mae"]
    rmse = por_clave["rmse"]
    r2 = por_clave["r2"]

    conclusiones = [
        "En MAE, el mejor resultado corresponde a "
        f"{mae['ganador']} con una diferencia absoluta de "
        f"{_formatear_moneda(mae['diferencia_absoluta'])}.",
        "En RMSE, el mejor resultado corresponde a "
        f"{rmse['ganador']} con una diferencia absoluta de "
        f"{_formatear_moneda(rmse['diferencia_absoluta'])}.",
        "En R2, el mejor resultado corresponde a "
        f"{r2['ganador']} con una diferencia de {r2['diferencia_absoluta']:.4f}.",
    ]

    if ganadores["global"] == "Empate":
        conclusiones.append(
            "La comparacion global queda equilibrada; ambos modelos muestran ventajas en metricas distintas."
        )
    else:
        conclusiones.append(
            f"El ganador global es {ganadores['global']}, porque obtiene ventaja en la mayoria de las metricas evaluadas."
        )

    if r2["random_forest"] < 0.4 and r2["red_neuronal"] < 0.4:
        conclusiones.append(
            "Ambos modelos tienen capacidad explicativa limitada; probablemente faltan variables relevantes como marca, version, estado o equipamiento."
        )
    elif r2["ganador"] == ganadores["global"]:
        conclusiones.append(
            "El modelo ganador tambien muestra mejor capacidad para explicar la variacion del precio."
        )

    return conclusiones


def generar_resumen_ejecutivo(ganadores: dict) -> str:
    """Resume el resultado global en una frase para la vista web."""
    if ganadores["global"] == "Empate":
        return (
            "La comparacion no muestra un dominio claro: cada modelo debe revisarse "
            "segun la metrica prioritaria del analisis."
        )
    return (
        f"{ganadores['global']} presenta el mejor balance general entre error y "
        "capacidad explicativa para este dataset."
    )


def obtener_contexto_comparacion_modelos() -> dict:
    """Obtiene cada contexto una sola vez y construye la comparacion completa."""
    contexto_random_forest = obtener_contexto_regresion()
    contexto_red_neuronal = obtener_contexto_red_neuronal_regresion()

    metricas_random_forest = contexto_random_forest["metricas"]
    metricas_red_neuronal = contexto_red_neuronal["metricas"]
    comparacion_metricas = construir_comparacion_metricas(
        metricas_random_forest,
        metricas_red_neuronal,
    )
    ganadores = determinar_ganadores(comparacion_metricas)
    conclusiones = generar_conclusiones_comparacion(comparacion_metricas, ganadores)

    return {
        "modelos": {
            "random_forest": contexto_random_forest["modelo"],
            "red_neuronal": contexto_red_neuronal["modelo"],
        },
        "registros_utilizados": {
            "random_forest": contexto_random_forest["registros_utilizados"],
            "red_neuronal": contexto_red_neuronal["registros_utilizados"],
        },
        "metricas_random_forest": metricas_random_forest,
        "metricas_red_neuronal": metricas_red_neuronal,
        "comparacion_metricas": comparacion_metricas,
        "ganadores": ganadores,
        "ganador_global": ganadores["global"],
        "conclusiones": conclusiones,
        "resumen_ejecutivo": generar_resumen_ejecutivo(ganadores),
        "chart_data": generar_chart_data_comparacion(comparacion_metricas),
    }
