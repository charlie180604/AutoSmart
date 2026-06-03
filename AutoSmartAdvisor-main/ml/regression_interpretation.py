"""Textos interpretativos para la vista web de regresion (sin alterar el modelo)."""

import numpy as np


def _formatear_moneda(valor: float) -> str:
    return f"${valor:,.0f}"


def generar_interpretacion_metricas(
    mae: float,
    rmse: float,
    r2: float,
    reales_vs_predichos: list[dict] | None = None,
) -> list[str]:
    """Genera lectura automatica de MAE, RMSE y R2 a partir de valores reales."""
    interpretaciones: list[str] = []

    interpretaciones.append(
        "En promedio, las estimaciones del modelo se desvian alrededor de "
        f"{_formatear_moneda(mae)} del precio real observado (error absoluto medio)."
    )

    if reales_vs_predichos:
        precios_reales = np.array(
            [float(item["precio_real"]) for item in reales_vs_predichos],
            dtype=float,
        )
        if precios_reales.size > 0:
            precio_medio = float(np.mean(precios_reales))
            if precio_medio > 0:
                ratio_mae = mae / precio_medio
                interpretaciones.append(
                    f"Ese error equivale aproximadamente al {ratio_mae * 100:.1f}% "
                    "del precio medio de los vehiculos evaluados en prueba."
                )
                if ratio_mae <= 0.15:
                    interpretaciones.append(
                        "En terminos relativos, la desviacion promedio es baja para apoyar "
                        "decisiones puntuales de compra."
                    )
                elif ratio_mae <= 0.35:
                    interpretaciones.append(
                        "En terminos relativos, la desviacion es moderada y conviene "
                        "complementar la estimacion con criterio de mercado."
                    )
                else:
                    interpretaciones.append(
                        "En terminos relativos, la desviacion es elevada; la estimacion "
                        "debe usarse como referencia general, no como valor exacto."
                    )

    ratio_rmse_mae = rmse / mae if mae > 0 else 1.0
    if ratio_rmse_mae >= 1.35:
        interpretaciones.append(
            f"El RMSE ({_formatear_moneda(rmse)}) supera de forma notable al MAE, "
            "lo que indica algunas predicciones con errores especialmente altos."
        )
    elif ratio_rmse_mae <= 1.1:
        interpretaciones.append(
            "El RMSE y el MAE son muy cercanos, lo que sugiere errores relativamente "
            "homogeneos entre los vehiculos evaluados."
        )
    else:
        interpretaciones.append(
            f"El RMSE ({_formatear_moneda(rmse)}) confirma que existen casos puntuales "
            "con desviaciones mayores al error tipico."
        )

    if r2 >= 0.7:
        interpretaciones.append(
            f"El coeficiente R² de {r2:.2f} indica una capacidad explicativa alta: "
            "el modelo reproduce buena parte de la variacion de precios en prueba."
        )
    elif r2 >= 0.4:
        interpretaciones.append(
            f"Con un R² de {r2:.2f}, el modelo explica una fraccion relevante de la "
            "variabilidad, aunque parte del precio sigue sin capturarse."
        )
    elif r2 >= 0.2:
        interpretaciones.append(
            f"El R² de {r2:.2f} es modesto: solo una porcion limitada de la variacion "
            "del precio queda explicada con las variables disponibles."
        )
    else:
        interpretaciones.append(
            f"El R² de {r2:.4f} es bajo, lo que sugiere que factores relevantes "
            "(marca, estado, equipamiento, ubicacion u otros) no estan en el dataset actual."
        )

    return interpretaciones


def generar_interpretacion_predicciones(reales_vs_predichos: list[dict]) -> list[str]:
    """Genera lectura del scatter real vs predicho."""
    if not reales_vs_predichos:
        return []

    reales = np.array(
        [float(item["precio_real"]) for item in reales_vs_predichos],
        dtype=float,
    )
    predichos = np.array(
        [float(item["precio_predicho"]) for item in reales_vs_predichos],
        dtype=float,
    )
    errores_abs = np.abs(reales - predichos)
    total = int(reales.size)

    interpretaciones = [
        f"Se analizaron {total} vehiculos del conjunto de prueba al comparar "
        "precio observado y precio estimado."
    ]

    p33, p66 = np.percentile(reales, [33.33, 66.67])
    en_rango_medio = int(np.sum((reales >= p33) & (reales <= p66)))
    pct_medio = 100.0 * en_rango_medio / total if total else 0.0
    if pct_medio >= 35:
        interpretaciones.append(
            f"Aproximadamente el {pct_medio:.0f}% de los vehiculos se concentra "
            f"en rangos medios de precio real (entre {_formatear_moneda(p33)} "
            f"y {_formatear_moneda(p66)})."
        )

    if errores_abs.size > 0 and np.mean(errores_abs) > 0:
        coef_variacion = float(np.std(errores_abs) / np.mean(errores_abs))
        if coef_variacion >= 0.85:
            interpretaciones.append(
                "La dispersion de los puntos es amplia: la calidad de la estimacion "
                "cambia de forma notable segun cada vehiculo."
            )
        else:
            interpretaciones.append(
                "La dispersion de los puntos es moderada, con agrupaciones visibles "
                "alrededor de la linea de referencia."
            )

    q25, q75 = np.percentile(reales, [25, 75])
    mascara_alto = reales >= q75
    mascara_bajo = reales <= q25
    if mascara_alto.any() and mascara_bajo.any():
        error_alto = float(np.mean(errores_abs[mascara_alto]))
        error_bajo = float(np.mean(errores_abs[mascara_bajo]))
        if error_alto > error_bajo * 1.2:
            interpretaciones.append(
                "Los errores tienden a aumentar en vehiculos de mayor valor: "
                f"error medio de {_formatear_moneda(error_alto)} en el cuartil superior "
                f"frente a {_formatear_moneda(error_bajo)} en el cuartil inferior."
            )
        elif error_bajo > error_alto * 1.2:
            interpretaciones.append(
                "El modelo presenta mayor desviacion en vehiculos de menor precio "
                "dentro del conjunto evaluado."
            )
        else:
            interpretaciones.append(
                "Los errores medios son comparables entre vehiculos de precio alto y bajo "
                "dentro del conjunto de prueba."
            )

    error_relativo = errores_abs / np.maximum(reales, 1.0)
    pct_cerca_diagonal = 100.0 * float(np.mean(error_relativo <= 0.25))
    if pct_cerca_diagonal >= 45:
        interpretaciones.append(
            f"Cerca del {pct_cerca_diagonal:.0f}% de los puntos queda dentro de un "
            "margen del 25% respecto al precio real, mostrando alineacion razonable "
            "con la linea de prediccion perfecta."
        )
    elif pct_cerca_diagonal < 25:
        interpretaciones.append(
            "Solo una fraccion reducida de los puntos se aproxima de forma cercana "
            "a la diagonal de referencia."
        )
    else:
        interpretaciones.append(
            f"Alrededor del {pct_cerca_diagonal:.0f}% de los puntos se acerca "
            "de forma moderada a la diagonal, aunque persisten desviaciones visibles."
        )

    return interpretaciones


def generar_interpretacion_errores(errores: list[dict]) -> list[str]:
    """Genera lectura del histograma de residuales."""
    if not errores:
        return []

    valores = np.array([float(item["error"]) for item in errores], dtype=float)
    absolutos = np.abs(valores)
    media = float(np.mean(valores))
    mediana_abs = float(np.median(absolutos))
    desviacion = float(np.std(valores))
    total = int(valores.size)

    interpretaciones = [
        f"El analisis de errores considera {total} residuales "
        "(precio real menos precio predicho) del conjunto de prueba."
    ]

    umbral_concentracion = float(np.percentile(absolutos, 40))
    pct_concentrados = 100.0 * float(np.mean(absolutos <= umbral_concentracion))
    if pct_concentrados >= 40:
        interpretaciones.append(
            f"La mayoria de los errores se concentra en magnitudes moderadas "
            f"(mediana absoluta de {_formatear_moneda(mediana_abs)})."
        )
    else:
        interpretaciones.append(
            f"Los errores se distribuyen de forma dispersa, con mediana absoluta "
            f"de {_formatear_moneda(mediana_abs)}."
        )

    percentil_90 = float(np.percentile(absolutos, 90))
    cantidad_extremos = int(np.sum(absolutos >= percentil_90))
    if cantidad_extremos > 0:
        interpretaciones.append(
            f"Existen vehiculos con errores elevados: el 10% superior supera "
            f"aproximadamente {_formatear_moneda(percentil_90)} de diferencia absoluta "
            f"({cantidad_extremos} casos en ese rango)."
        )

    precios_reales = np.array(
        [float(item["precio_real"]) for item in errores],
        dtype=float,
    )
    umbral_sesgo = max(20_000.0, 0.04 * float(np.mean(precios_reales)) if precios_reales.size else 20_000.0)

    if media > umbral_sesgo:
        interpretaciones.append(
            "El modelo tiende a subestimar el precio: los valores reales suelen ser "
            f"mayores que los predichos (error promedio de {_formatear_moneda(media)})."
        )
    elif media < -umbral_sesgo:
        interpretaciones.append(
            "El modelo tiende a sobreestimar el precio: los valores predichos suelen "
            f"ser mayores que los reales (error promedio de {_formatear_moneda(media)})."
        )
    else:
        interpretaciones.append(
            "No se observa un sesgo marcado hacia sobreestimacion o subestimacion: "
            f"el error promedio es {_formatear_moneda(media)}."
        )

    if desviacion > mediana_abs * 1.1 and mediana_abs > 0:
        interpretaciones.append(
            f"La dispersion de los errores es amplia (desviacion de {_formatear_moneda(desviacion)}), "
            "lo que refleja estimaciones heterogeneas entre vehiculos."
        )

    return interpretaciones


def enriquecer_contexto_regresion(contexto: dict) -> dict:
    """Agrega interpretaciones a partir del contexto ya calculado por regression_service."""
    chart_data = contexto.get("chart_data") or {}
    reales_vs_predichos = chart_data.get("reales_vs_predichos") or []
    errores = chart_data.get("errores") or []

    contexto["interpretacion_metricas"] = generar_interpretacion_metricas(
        float(contexto["mae"]),
        float(contexto["rmse"]),
        float(contexto["r2"]),
        reales_vs_predichos,
    )
    contexto["interpretacion_predicciones"] = generar_interpretacion_predicciones(
        reales_vs_predichos
    )
    contexto["interpretacion_errores"] = generar_interpretacion_errores(errores)
    return contexto
