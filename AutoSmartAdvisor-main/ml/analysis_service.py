"""Servicio de datos para la pagina de analisis exploratorio."""

from ml.exploratory_analysis import (
    calcular_estadisticas_descriptivas,
    calcular_matriz_correlacion,
    cargar_dataset,
    detectar_outliers_principales,
    generar_resumen_exploratorio,
    obtener_variables_numericas_limpias,
)


def _formatear_moneda(valor):
    if valor is None:
        return "No disponible"
    return f"${valor:,.0f}"


def _formatear_numero(valor, decimales=0):
    if valor is None:
        return "No disponible"
    return f"{valor:,.{decimales}f}"


def _serie_como_lista(datos, columna):
    if columna not in datos.columns:
        return []
    return datos[columna].dropna().round(2).tolist()


def _pares_como_listas(datos, columna_x, columna_y):
    if columna_x not in datos.columns or columna_y not in datos.columns:
        return [], []
    pares = datos[[columna_x, columna_y]].dropna()
    return pares[columna_x].round(2).tolist(), pares[columna_y].round(2).tolist()


def _buscar_correlacion(correlacion, fila, columna):
    columnas = correlacion.get("columnas", [])
    matriz = correlacion.get("matriz", [])
    if fila not in columnas or columna not in columnas:
        return None

    indice_fila = columnas.index(fila)
    valor = matriz[indice_fila].get(columna)
    return None if valor != valor else valor


def _limpiar_valor_correlacion(valor):
    return None if valor != valor else valor


def _interpretar_correlacion(nombre, valor):
    if valor is None:
        return {
            "nombre": nombre,
            "valor": "No disponible",
            "clase": "neutral",
            "texto": "No hay datos suficientes para interpretar esta relacion.",
        }

    if valor >= 0.3:
        texto = "La relacion es positiva: cuando una variable sube, la otra suele aumentar."
        clase = "positiva"
    elif valor <= -0.3:
        texto = "La relacion es negativa: cuando una variable sube, la otra suele disminuir."
        clase = "negativa"
    else:
        texto = "La relacion es baja: no se observa un movimiento fuerte entre ambas variables."
        clase = "neutral"

    return {
        "nombre": nombre,
        "valor": f"{valor:.2f}",
        "clase": clase,
        "texto": texto,
    }


def obtener_contexto_analisis():
    """Construye el contexto visual de /analisis desde exploratory_analysis.py."""
    autos = cargar_dataset()
    resumen = generar_resumen_exploratorio(autos)
    estadisticas = calcular_estadisticas_descriptivas(autos)
    outliers = detectar_outliers_principales(autos)
    correlacion = calcular_matriz_correlacion(
        autos,
        columnas=("precio", "anio", "kilometraje"),
    )
    datos_numericos = obtener_variables_numericas_limpias(
        autos,
        columnas=("precio", "anio", "kilometraje", "pasajeros"),
        eliminar_nulos=False,
    )

    precio = estadisticas["precio"]
    anio = estadisticas["anio"]
    kilometraje = estadisticas["kilometraje"]

    anios_precio, precios_por_anio = _pares_como_listas(datos_numericos, "anio", "precio")
    kilometrajes, precios_por_km = _pares_como_listas(
        datos_numericos,
        "kilometraje",
        "precio",
    )

    precio_anio = _buscar_correlacion(correlacion, "precio", "anio")
    precio_kilometraje = _buscar_correlacion(correlacion, "precio", "kilometraje")
    anio_kilometraje = _buscar_correlacion(correlacion, "anio", "kilometraje")

    return {
        "resumen": resumen,
        "metricas": [
            {
                "titulo": "Total de vehiculos",
                "valor": _formatear_numero(resumen["validacion"]["registros"]),
                "icono": "bi-car-front",
            },
            {
                "titulo": "Precio promedio",
                "valor": _formatear_moneda(precio["promedio"]),
                "icono": "bi-cash-stack",
            },
            {
                "titulo": "Precio mediano",
                "valor": _formatear_moneda(precio["mediana"]),
                "icono": "bi-bar-chart",
            },
            {
                "titulo": "Precio minimo",
                "valor": _formatear_moneda(precio["minimo"]),
                "icono": "bi-arrow-down-circle",
            },
            {
                "titulo": "Precio maximo",
                "valor": _formatear_moneda(precio["maximo"]),
                "icono": "bi-arrow-up-circle",
            },
            {
                "titulo": "Anio promedio",
                "valor": _formatear_numero(anio["promedio"]),
                "icono": "bi-calendar3",
            },
            {
                "titulo": "Kilometraje promedio",
                "valor": f"{_formatear_numero(kilometraje['promedio'])} km",
                "icono": "bi-speedometer2",
            },
        ],
        "outliers": {
            "precio": {
                **outliers["precio"],
                "minimo_formateado": _formatear_moneda(
                    outliers["precio"]["valor_minimo_atipico"]
                ),
                "maximo_formateado": _formatear_moneda(
                    outliers["precio"]["valor_maximo_atipico"]
                ),
            },
            "kilometraje": {
                **outliers["kilometraje"],
                "minimo_formateado": _formatear_numero(
                    outliers["kilometraje"]["valor_minimo_atipico"]
                ),
                "maximo_formateado": _formatear_numero(
                    outliers["kilometraje"]["valor_maximo_atipico"]
                ),
            },
        },
        "correlaciones": [
            _interpretar_correlacion("Precio - Anio", precio_anio),
            _interpretar_correlacion("Precio - Kilometraje", precio_kilometraje),
            _interpretar_correlacion("Anio - Kilometraje", anio_kilometraje),
        ],
        "chart_data": {
            "precios": _serie_como_lista(datos_numericos, "precio"),
            "anios_precio": anios_precio,
            "precios_por_anio": precios_por_anio,
            "kilometrajes": kilometrajes,
            "precios_por_kilometraje": precios_por_km,
            "correlacion_columnas": correlacion["columnas"],
            "correlacion_matriz": [
                [
                    _limpiar_valor_correlacion(fila.get(columna))
                    for columna in correlacion["columnas"]
                ]
                for fila in correlacion["matriz"]
            ],
        },
    }
