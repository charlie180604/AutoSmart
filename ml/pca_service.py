"""Servicio reutilizable para Analisis de Componentes Principales (PCA)."""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ml.exploratory_analysis import (
    cargar_dataset,
    obtener_variables_numericas_limpias,
)


VARIABLES_CANDIDATAS_PCA = (
    "precio",
    "anio",
    "kilometraje",
    "rendimiento",
    "pasajeros",
)


def _redondear_lista(valores, decimales=4):
    return [round(float(valor), decimales) for valor in valores]


def _filtrar_variables_utiles(datos):
    """Elimina columnas constantes o sin variabilidad util para PCA."""
    columnas_utiles = [
        columna for columna in datos.columns if datos[columna].nunique(dropna=True) > 1
    ]
    return datos[columnas_utiles].copy()


def preparar_datos_pca(autos=None):
    """Carga, selecciona y normaliza variables numericas utiles para PCA."""
    if autos is None:
        autos = cargar_dataset()

    datos_numericos = obtener_variables_numericas_limpias(
        autos,
        columnas=VARIABLES_CANDIDATAS_PCA,
        eliminar_nulos=True,
    )
    datos_utiles = _filtrar_variables_utiles(datos_numericos)

    if datos_utiles.empty:
        raise ValueError("No hay variables numericas utiles para aplicar PCA.")

    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos_utiles)
    datos_normalizados = pd.DataFrame(
        datos_escalados,
        columns=datos_utiles.columns,
        index=datos_utiles.index,
    )
    evidencia_escalado = {
        "promedios": {
            columna: round(float(valor), 6)
            for columna, valor in datos_normalizados.mean().to_dict().items()
        },
        "desviaciones_estandar": {
            columna: round(float(valor), 6)
            for columna, valor in datos_normalizados.std(ddof=0).to_dict().items()
        },
    }

    return {
        "datos_originales": datos_utiles,
        "datos_normalizados": datos_normalizados,
        "variables_utilizadas": list(datos_utiles.columns),
        "cantidad_registros": int(len(datos_utiles)),
        "evidencia_escalado": evidencia_escalado,
    }


def aplicar_pca(datos_normalizados):
    """Ejecuta PCA con scikit-learn y devuelve resultados reutilizables."""
    cantidad_componentes = min(datos_normalizados.shape)
    pca = PCA(n_components=cantidad_componentes)
    componentes = pca.fit_transform(datos_normalizados)
    varianza_explicada = pca.explained_variance_ratio_
    varianza_acumulada = varianza_explicada.cumsum()
    nombres_componentes = [
        f"PC{indice + 1}" for indice in range(cantidad_componentes)
    ]

    return {
        "modelo": pca,
        "componentes": componentes,
        "nombres_componentes": nombres_componentes,
        "varianza_explicada": _redondear_lista(varianza_explicada),
        "varianza_explicada_porcentaje": _redondear_lista(
            varianza_explicada * 100,
            decimales=2,
        ),
        "varianza_acumulada": _redondear_lista(varianza_acumulada),
        "varianza_acumulada_porcentaje": _redondear_lista(
            varianza_acumulada * 100,
            decimales=2,
        ),
        "cantidad_componentes": int(cantidad_componentes),
        "cargas_componentes": {
            componente: {
                columna: round(float(valor), 4)
                for columna, valor in zip(datos_normalizados.columns, cargas)
            }
            for componente, cargas in zip(nombres_componentes, pca.components_)
        },
    }


def generar_conclusiones_pca(resultado_pca):
    """Genera interpretaciones automaticas a partir de resultados reales."""
    varianza_acumulada = resultado_pca["varianza_acumulada_porcentaje"]
    cantidad_componentes = resultado_pca["cantidad_componentes"]

    conclusiones = []
    if not varianza_acumulada:
        return conclusiones

    if cantidad_componentes >= 2:
        conclusiones.append(
            "Los primeros dos componentes conservan el "
            f"{varianza_acumulada[1]:.2f}% de la informacion del dataset."
        )
    else:
        conclusiones.append(
            "El primer componente conserva el "
            f"{varianza_acumulada[0]:.2f}% de la informacion del dataset."
        )

    componentes_80 = next(
        (
            indice + 1
            for indice, valor in enumerate(varianza_acumulada)
            if valor >= 80
        ),
        cantidad_componentes,
    )
    conclusiones.append(
        "Se necesitan "
        f"{componentes_80} componente(s) para conservar al menos el 80% "
        "de la variabilidad."
    )

    if cantidad_componentes > 2 and varianza_acumulada[1] >= 70:
        conclusiones.append(
            "El dataset puede representarse en dos dimensiones manteniendo "
            "una parte importante de su estructura."
        )
    else:
        conclusiones.append(
            "La reduccion de dimensionalidad debe revisarse con cuidado para "
            "no perder informacion relevante."
        )

    return conclusiones


def preparar_datos_visualizacion(resultado_pca):
    """Prepara estructuras listas para graficas de PCA sin generar HTML."""
    componentes = resultado_pca["componentes"]
    nombres = resultado_pca["nombres_componentes"]
    varianza = resultado_pca["varianza_explicada_porcentaje"]
    acumulada = resultado_pca["varianza_acumulada_porcentaje"]

    pc1 = componentes[:, 0].round(4).tolist()
    pc2 = (
        componentes[:, 1].round(4).tolist()
        if resultado_pca["cantidad_componentes"] >= 2
        else [0 for _ in pc1]
    )

    return {
        "componentes": nombres,
        "varianza_explicada": varianza,
        "varianza_acumulada": acumulada,
        "pca_2d": {
            "pc1": pc1,
            "pc2": pc2,
        },
    }


def obtener_contexto_pca():
    """Funcion central para construir todo el contexto de la pagina /pca."""
    preparacion = preparar_datos_pca()
    resultado_pca = aplicar_pca(preparacion["datos_normalizados"])
    conclusiones = generar_conclusiones_pca(resultado_pca)
    chart_data = preparar_datos_visualizacion(resultado_pca)

    return {
        "variables_utilizadas": preparacion["variables_utilizadas"],
        "cantidad_registros": preparacion["cantidad_registros"],
        "evidencia_escalado": preparacion["evidencia_escalado"],
        "componentes_principales": resultado_pca["nombres_componentes"],
        "cantidad_componentes": resultado_pca["cantidad_componentes"],
        "varianza_explicada": resultado_pca["varianza_explicada_porcentaje"],
        "varianza_acumulada": resultado_pca["varianza_acumulada_porcentaje"],
        "cargas_componentes": resultado_pca["cargas_componentes"],
        "conclusiones": conclusiones,
        "chart_data": chart_data,
    }
