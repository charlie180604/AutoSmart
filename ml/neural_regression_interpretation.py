"""Interpretaciones automaticas para regresion con red neuronal."""

import numpy as np


def _formatear_moneda(valor: float) -> str:
    return f"${valor:,.0f}"


def _limitar_observaciones(observaciones: list[str], minimo: int = 4, maximo: int = 6) -> list[str]:
    if len(observaciones) >= minimo:
        return observaciones[:maximo]

    relleno = [
        "La lectura debe complementarse con las graficas y con el contexto del mercado automotriz.",
        "Los resultados representan el comportamiento del conjunto de prueba, no una garantia para cada vehiculo individual.",
        "La red neuronal aprende patrones generales del dataset disponible y puede cambiar si se agregan nuevas variables.",
    ]
    for texto in relleno:
        if len(observaciones) >= minimo:
            break
        observaciones.append(texto)
    return observaciones[:maximo]


def generar_interpretacion_metricas(
    mae: float,
    rmse: float,
    r2: float,
    reales_vs_predichos: list[dict] | None = None,
) -> list[str]:
    """Genera observaciones sobre MAE, RMSE y R2 en escala real de precio."""
    observaciones = [
        "En promedio, la red neuronal se desvia alrededor de "
        f"{_formatear_moneda(mae)} respecto al precio real observado.",
    ]

    if reales_vs_predichos:
        precios_reales = np.array(
            [float(item["precio_real"]) for item in reales_vs_predichos],
            dtype=float,
        )
        if precios_reales.size:
            precio_medio = float(np.mean(precios_reales))
            if precio_medio > 0:
                ratio_mae = mae / precio_medio
                observaciones.append(
                    f"El error promedio equivale aproximadamente al {ratio_mae * 100:.1f}% "
                    "del precio medio de los vehiculos evaluados."
                )
                if ratio_mae <= 0.15:
                    observaciones.append(
                        "En terminos relativos, el nivel de error es bajo para una estimacion inicial."
                    )
                elif ratio_mae <= 0.35:
                    observaciones.append(
                        "En terminos relativos, el nivel de error es moderado y conviene usarlo como referencia."
                    )
                else:
                    observaciones.append(
                        "En terminos relativos, el error es elevado; la prediccion debe tomarse como orientativa."
                    )

    ratio_rmse_mae = rmse / mae if mae > 0 else 1.0
    if ratio_rmse_mae >= 1.35:
        observaciones.append(
            f"El RMSE ({_formatear_moneda(rmse)}) es bastante mayor que el MAE, "
            "lo que sugiere algunos errores grandes en casos especificos."
        )
    elif ratio_rmse_mae <= 1.1:
        observaciones.append(
            "El RMSE y el MAE son cercanos, senal de errores relativamente estables."
        )
    else:
        observaciones.append(
            f"El RMSE ({_formatear_moneda(rmse)}) confirma que existen desviaciones "
            "mayores al error promedio en parte del conjunto de prueba."
        )

    if r2 >= 0.7:
        observaciones.append(
            f"El R2 de {r2:.2f} indica que la red captura una parte alta de la variacion del precio."
        )
    elif r2 >= 0.4:
        observaciones.append(
            f"El R2 de {r2:.2f} muestra una capacidad explicativa intermedia."
        )
    elif r2 >= 0.2:
        observaciones.append(
            f"El R2 de {r2:.2f} es modesto: la red aprende patrones utiles, pero limitados."
        )
    else:
        observaciones.append(
            f"El R2 de {r2:.4f} es bajo, probablemente porque faltan variables relevantes "
            "como marca, version, estado fisico o equipamiento."
        )

    return _limitar_observaciones(observaciones)


def generar_interpretacion_predicciones(reales_vs_predichos: list[dict]) -> list[str]:
    """Genera observaciones sobre la relacion entre precios reales y predichos."""
    if not reales_vs_predichos:
        return _limitar_observaciones([])

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

    observaciones = [
        f"Se compararon {total} vehiculos del conjunto de prueba entre precio real y precio estimado.",
    ]

    if total:
        correlacion = float(np.corrcoef(reales, predichos)[0, 1]) if np.std(predichos) > 0 else 0.0
        if correlacion >= 0.75:
            observaciones.append(
                f"La relacion entre precios reales y predichos es alta (correlacion aproximada de {correlacion:.2f})."
            )
        elif correlacion >= 0.45:
            observaciones.append(
                f"La relacion entre precios reales y predichos es moderada (correlacion aproximada de {correlacion:.2f})."
            )
        else:
            observaciones.append(
                f"La relacion entre precios reales y predichos es baja (correlacion aproximada de {correlacion:.2f})."
            )

    coef_variacion = float(np.std(errores_abs) / np.mean(errores_abs)) if np.mean(errores_abs) > 0 else 0.0
    if coef_variacion >= 0.85:
        observaciones.append(
            "La dispersion de las predicciones es amplia: algunos vehiculos se estiman mucho mejor que otros."
        )
    else:
        observaciones.append(
            "La dispersion de las predicciones es moderada y mantiene cierta consistencia entre casos."
        )

    q25, q75 = np.percentile(reales, [25, 75])
    mascara_alto = reales >= q75
    mascara_bajo = reales <= q25
    if mascara_alto.any() and mascara_bajo.any():
        error_alto = float(np.mean(errores_abs[mascara_alto]))
        error_bajo = float(np.mean(errores_abs[mascara_bajo]))
        if error_alto > error_bajo * 1.2:
            observaciones.append(
                "La red muestra mayor dificultad en vehiculos de precio alto: "
                f"error medio de {_formatear_moneda(error_alto)} frente a "
                f"{_formatear_moneda(error_bajo)} en precios bajos."
            )
        elif error_bajo > error_alto * 1.2:
            observaciones.append(
                "La red muestra mayor dificultad en vehiculos de precio bajo dentro del conjunto evaluado."
            )
        else:
            observaciones.append(
                "El error medio es parecido entre vehiculos de precio alto y bajo."
            )

    error_relativo = errores_abs / np.maximum(reales, 1.0)
    pct_cerca = 100.0 * float(np.mean(error_relativo <= 0.25))
    if pct_cerca >= 45:
        observaciones.append(
            f"Cerca del {pct_cerca:.0f}% de las predicciones queda dentro de un margen del 25% del precio real."
        )
    elif pct_cerca >= 25:
        observaciones.append(
            f"Alrededor del {pct_cerca:.0f}% de las predicciones se acerca razonablemente al precio real."
        )
    else:
        observaciones.append(
            f"Solo cerca del {pct_cerca:.0f}% de las predicciones queda dentro de un margen del 25%."
        )

    return _limitar_observaciones(observaciones)


def generar_interpretacion_errores(errores: list[dict]) -> list[str]:
    """Genera observaciones sobre residuales y tendencia de error."""
    if not errores:
        return _limitar_observaciones([])

    valores = np.array([float(item["error"]) for item in errores], dtype=float)
    precios_reales = np.array([float(item["precio_real"]) for item in errores], dtype=float)
    absolutos = np.abs(valores)
    total = int(valores.size)
    media = float(np.mean(valores))
    mediana_abs = float(np.median(absolutos))
    desviacion = float(np.std(valores))

    observaciones = [
        f"El analisis considera {total} errores, calculados como precio real menos precio predicho.",
        f"La mediana del error absoluto es {_formatear_moneda(mediana_abs)}, lo que resume el error tipico sin depender tanto de casos extremos.",
    ]

    percentil_90 = float(np.percentile(absolutos, 90))
    cantidad_extremos = int(np.sum(absolutos >= percentil_90))
    observaciones.append(
        f"El 10% de errores mas altos supera aproximadamente {_formatear_moneda(percentil_90)} "
        f"({cantidad_extremos} vehiculos en ese rango)."
    )

    umbral_sesgo = max(
        20_000.0,
        0.04 * float(np.mean(precios_reales)) if precios_reales.size else 20_000.0,
    )
    if media > umbral_sesgo:
        observaciones.append(
            "La red tiende a subestimar precios: los valores reales quedan por encima "
            f"de los predichos en promedio ({_formatear_moneda(media)})."
        )
    elif media < -umbral_sesgo:
        observaciones.append(
            "La red tiende a sobreestimar precios: los valores predichos quedan por encima "
            f"de los reales en promedio ({_formatear_moneda(media)})."
        )
    else:
        observaciones.append(
            "No se observa un sesgo fuerte hacia sobreestimacion o subestimacion "
            f"(error promedio de {_formatear_moneda(media)})."
        )

    if mediana_abs > 0 and desviacion > mediana_abs * 1.5:
        observaciones.append(
            f"La dispersion de errores es amplia (desviacion de {_formatear_moneda(desviacion)}), "
            "por lo que la estabilidad cambia segun el vehiculo."
        )
    else:
        observaciones.append(
            "La dispersion de errores es relativamente controlada para el conjunto evaluado."
        )

    return _limitar_observaciones(observaciones)


def generar_interpretacion_entrenamiento(historial_entrenamiento: dict) -> list[str]:
    """Genera observaciones sobre loss, val_loss y EarlyStopping."""
    loss = np.array(historial_entrenamiento.get("loss") or [], dtype=float)
    val_loss = np.array(historial_entrenamiento.get("val_loss") or [], dtype=float)
    epochs = int(loss.size)

    if epochs == 0:
        return _limitar_observaciones([])

    observaciones = [
        f"La red neuronal entreno durante {epochs} epochs antes de conservar sus mejores pesos.",
    ]

    if epochs < 300:
        observaciones.append(
            "El entrenamiento se detuvo antes del maximo configurado, senal de que EarlyStopping encontro un punto adecuado para evitar sobreajuste."
        )
    else:
        observaciones.append(
            "El entrenamiento uso todas las epochs configuradas, por lo que conviene revisar si la perdida aun seguia mejorando."
        )

    reduccion_loss = float(loss[0] - loss[-1]) if loss.size else 0.0
    if loss[0] != 0:
        pct_reduccion = 100.0 * reduccion_loss / abs(float(loss[0]))
        observaciones.append(
            f"La perdida de entrenamiento cambio de {loss[0]:.4f} a {loss[-1]:.4f}, "
            f"una reduccion aproximada de {pct_reduccion:.1f}%."
        )

    if val_loss.size:
        mejor_epoch = int(np.argmin(val_loss) + 1)
        observaciones.append(
            f"La mejor perdida de validacion se observo alrededor de la epoch {mejor_epoch}, "
            f"con val_loss de {float(np.min(val_loss)):.4f}."
        )

        brecha_final = float(val_loss[-1] - loss[-1])
        if brecha_final > max(0.05, abs(float(loss[-1])) * 0.25):
            observaciones.append(
                "La diferencia final entre entrenamiento y validacion sugiere posible sobreajuste en las ultimas epochs."
            )
        else:
            observaciones.append(
                "La perdida de validacion se mantiene cercana a la de entrenamiento, senal de un aprendizaje relativamente estable."
            )

    if loss.size >= 5:
        ultimos = loss[-5:]
        variacion_final = float(np.std(ultimos))
        if variacion_final <= 0.02:
            observaciones.append(
                "Las ultimas epochs muestran cambios pequenos en la perdida, por lo que el entrenamiento parece estabilizado."
            )
        else:
            observaciones.append(
                "Las ultimas epochs aun muestran variacion en la perdida, aunque EarlyStopping ayuda a seleccionar el mejor punto."
            )

    return _limitar_observaciones(observaciones)


def enriquecer_contexto_red_neuronal(contexto: dict) -> dict:
    """Agrega interpretaciones automaticas al contexto de neural_regression_service."""
    chart_data = contexto.get("chart_data") or {}
    reales_vs_predichos = chart_data.get("reales_vs_predichos") or []
    errores = chart_data.get("errores") or []
    historial = contexto.get("historial_entrenamiento") or {}

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
    contexto["interpretacion_entrenamiento"] = generar_interpretacion_entrenamiento(
        historial
    )
    return contexto
