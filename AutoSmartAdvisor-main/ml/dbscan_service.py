"""Servicio backend para clustering con DBSCAN.

Esta fase prepara DBSCAN sin modificar la interfaz ni el recomendador. El
flujo reutiliza el dataset limpio, variables numericas utiles, StandardScaler
y PCA antes del agrupamiento por densidad.
"""

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from ml.exploratory_analysis import (
    cargar_dataset,
    obtener_variables_numericas_limpias,
)
from ml.pca_service import VARIABLES_CANDIDATAS_PCA, aplicar_pca


DEFAULT_EPS = 0.8
DEFAULT_MIN_SAMPLES = 8
EPS_VALUES_TO_EVALUATE = (0.4, 0.6, 0.8, 1.0, 1.2)


def _filtrar_variables_utiles(datos: pd.DataFrame) -> pd.DataFrame:
    """Elimina variables constantes o sin variabilidad util para DBSCAN."""
    columnas_utiles = [
        columna for columna in datos.columns if datos[columna].nunique(dropna=True) > 1
    ]
    return datos[columnas_utiles].copy()


def preparar_datos_dbscan(autos=None) -> dict:
    """Carga, limpia y escala variables numericas utiles para DBSCAN."""
    if autos is None:
        autos = cargar_dataset()

    datos_numericos = obtener_variables_numericas_limpias(
        autos,
        columnas=VARIABLES_CANDIDATAS_PCA,
        eliminar_nulos=True,
    )
    datos_utiles = _filtrar_variables_utiles(datos_numericos)

    if datos_utiles.empty:
        raise ValueError("No hay variables numericas utiles para aplicar DBSCAN.")

    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos_utiles)
    datos_escalados_df = pd.DataFrame(
        datos_escalados,
        columns=datos_utiles.columns,
        index=datos_utiles.index,
    )

    evidencia_escalado = {
        "promedios": {
            columna: round(float(valor), 6)
            for columna, valor in datos_escalados_df.mean().to_dict().items()
        },
        "desviaciones_estandar": {
            columna: round(float(valor), 6)
            for columna, valor in datos_escalados_df.std(ddof=0).to_dict().items()
        },
    }

    return {
        "datos_originales": datos_utiles,
        "datos_escalados": datos_escalados_df,
        "variables_utilizadas": list(datos_utiles.columns),
        "cantidad_registros": int(len(datos_utiles)),
        "evidencia_escalado": evidencia_escalado,
    }


def aplicar_pca_para_dbscan(datos_escalados: pd.DataFrame) -> dict:
    """Aplica el PCA ya implementado antes del clustering por densidad."""
    resultado_pca = aplicar_pca(datos_escalados)
    componentes_df = pd.DataFrame(
        resultado_pca["componentes"],
        columns=resultado_pca["nombres_componentes"],
        index=datos_escalados.index,
    )

    return {
        **resultado_pca,
        "componentes_df": componentes_df,
    }


def ejecutar_dbscan(
    datos_pca: pd.DataFrame,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict:
    """Ejecuta DBSCAN sobre los componentes PCA."""
    if eps <= 0:
        raise ValueError("eps debe ser mayor que 0.")
    if min_samples < 1:
        raise ValueError("min_samples debe ser al menos 1.")

    modelo = DBSCAN(eps=eps, min_samples=min_samples)
    etiquetas = modelo.fit_predict(datos_pca)

    return {
        "modelo": modelo,
        "etiquetas": etiquetas,
        "eps": float(eps),
        "min_samples": int(min_samples),
    }


def identificar_clusters_y_ruido(etiquetas) -> dict:
    """Identifica clusters validos y registros marcados como ruido (-1)."""
    etiquetas_serie = pd.Series(etiquetas, name="cluster")
    clusters = sorted(
        int(etiqueta)
        for etiqueta in etiquetas_serie.unique()
        if int(etiqueta) != -1
    )
    cantidad_ruido = int((etiquetas_serie == -1).sum())
    total = int(len(etiquetas_serie))

    return {
        "clusters": clusters,
        "cantidad_clusters": int(len(clusters)),
        "cantidad_ruido": cantidad_ruido,
        "porcentaje_ruido": round((cantidad_ruido / total) * 100, 2),
        "total_registros": total,
    }


def obtener_distribucion_clusters_dbscan(etiquetas) -> dict:
    """Devuelve distribucion de clusters, incluyendo ruido como -1."""
    serie = pd.Series(etiquetas, name="cluster")
    distribucion = serie.value_counts().sort_index()
    total = int(distribucion.sum())

    return {
        int(cluster): {
            "cantidad": int(cantidad),
            "porcentaje": round((int(cantidad) / total) * 100, 2),
            "tipo": "ruido" if int(cluster) == -1 else "cluster",
        }
        for cluster, cantidad in distribucion.items()
    }


def calcular_metricas_dbscan(datos_pca: pd.DataFrame, etiquetas) -> dict:
    """Calcula metricas compatibles cuando DBSCAN produce clusters validos."""
    etiquetas_serie = pd.Series(etiquetas)
    mascara_clusters = etiquetas_serie != -1
    datos_clusterizados = datos_pca.loc[mascara_clusters.values]
    etiquetas_clusterizadas = etiquetas_serie[mascara_clusters].values
    cantidad_clusters = len(set(etiquetas_clusterizadas))

    if cantidad_clusters < 2:
        return {
            "silhouette_score": None,
            "calinski_harabasz_score": None,
            "davies_bouldin_score": None,
            "mensaje": "Las metricas requieren al menos 2 clusters sin contar ruido.",
        }

    return {
        "silhouette_score": round(
            float(silhouette_score(datos_clusterizados, etiquetas_clusterizadas)),
            4,
        ),
        "calinski_harabasz_score": round(
            float(calinski_harabasz_score(datos_clusterizados, etiquetas_clusterizadas)),
            4,
        ),
        "davies_bouldin_score": round(
            float(davies_bouldin_score(datos_clusterizados, etiquetas_clusterizadas)),
            4,
        ),
        "mensaje": "Metricas calculadas excluyendo registros marcados como ruido.",
    }


def evaluar_valores_eps(
    datos_pca: pd.DataFrame,
    valores_eps=EPS_VALUES_TO_EVALUATE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> list[dict]:
    """Evalua diferentes valores de eps sin cambiar el DBSCAN principal."""
    evaluaciones = []
    for eps in valores_eps:
        resultado = ejecutar_dbscan(
            datos_pca,
            eps=eps,
            min_samples=min_samples,
        )
        resumen = identificar_clusters_y_ruido(resultado["etiquetas"])
        evaluaciones.append(
            {
                "eps": float(eps),
                "cantidad_clusters": resumen["cantidad_clusters"],
                "cantidad_ruido": resumen["cantidad_ruido"],
                "porcentaje_ruido": resumen["porcentaje_ruido"],
            }
        )

    return evaluaciones


def generar_conclusiones_dbscan(
    resumen_clusters: dict,
    distribucion_clusters: dict,
    eps: float,
    min_samples: int,
    evaluacion_eps: list[dict] | None = None,
) -> list[str]:
    """Genera conclusiones automaticas basadas en resultados reales."""
    cantidad_clusters = resumen_clusters["cantidad_clusters"]
    cantidad_ruido = resumen_clusters["cantidad_ruido"]
    porcentaje_ruido = resumen_clusters["porcentaje_ruido"]

    conclusiones = [
        f"DBSCAN identifico {cantidad_clusters} grupo(s) densos con eps={eps} y min_samples={min_samples}.",
        f"Se detectaron {cantidad_ruido} registros como ruido, equivalentes al {porcentaje_ruido}% del dataset.",
    ]

    if porcentaje_ruido <= 5:
        conclusiones.append(
            "El porcentaje de ruido es bajo respecto al total del dataset."
        )
    elif porcentaje_ruido <= 20:
        conclusiones.append(
            "El porcentaje de ruido es moderado y puede representar casos atipicos relevantes."
        )
    else:
        conclusiones.append(
            "El porcentaje de ruido es alto; conviene revisar eps y min_samples en futuras pruebas."
        )

    clusters_validos = {
        cluster: datos
        for cluster, datos in distribucion_clusters.items()
        if cluster != -1
    }
    if clusters_validos:
        cluster_mayor = max(
            clusters_validos,
            key=lambda cluster: clusters_validos[cluster]["cantidad"],
        )
        conclusiones.append(
            "El cluster mas grande es el "
            f"{cluster_mayor} con "
            f"{clusters_validos[cluster_mayor]['cantidad']} registros."
        )
    else:
        conclusiones.append(
            "No se identificaron clusters densos; todos los registros quedaron como ruido."
        )

    if evaluacion_eps:
        menor_eps = min(evaluacion_eps, key=lambda item: item["eps"])
        mayor_eps = max(evaluacion_eps, key=lambda item: item["eps"])
        mejor_ruido = min(
            evaluacion_eps,
            key=lambda item: item["porcentaje_ruido"],
        )
        conclusiones.append(
            "Al aumentar eps de "
            f"{menor_eps['eps']} a {mayor_eps['eps']}, el porcentaje de ruido "
            f"cambia de {menor_eps['porcentaje_ruido']}% a "
            f"{mayor_eps['porcentaje_ruido']}%."
        )
        conclusiones.append(
            "Dentro de los valores evaluados, eps="
            f"{mejor_ruido['eps']} produjo el menor porcentaje de ruido "
            f"({mejor_ruido['porcentaje_ruido']}%)."
        )

    return conclusiones


def preparar_datos_visualizacion_dbscan(
    datos_originales: pd.DataFrame,
    datos_pca: pd.DataFrame,
    etiquetas,
    distribucion_clusters: dict,
    evaluacion_eps: list[dict],
    metricas: dict,
) -> dict:
    """Prepara estructuras listas para graficas Plotly de DBSCAN."""
    datos_visuales = pd.DataFrame(
        {
            "PC1": datos_pca["PC1"].round(4),
            "PC2": datos_pca["PC2"].round(4),
            "cluster": [int(etiqueta) for etiqueta in etiquetas],
            "es_ruido": [int(etiqueta) == -1 for etiqueta in etiquetas],
        },
        index=datos_pca.index,
    )

    for columna in datos_originales.columns:
        datos_visuales[columna] = datos_originales[columna].round(2)

    ruido = datos_visuales[datos_visuales["es_ruido"]]

    return {
        "pca_2d_clusters": datos_visuales.to_dict(orient="records"),
        "ruido": ruido.to_dict(orient="records"),
        "distribucion_clusters": {
            "clusters": [
                int(cluster) for cluster in distribucion_clusters.keys()
            ],
            "etiquetas": [
                "Ruido" if int(cluster) == -1 else f"Cluster {cluster}"
                for cluster in distribucion_clusters.keys()
            ],
            "cantidades": [
                datos_cluster["cantidad"]
                for datos_cluster in distribucion_clusters.values()
            ],
            "porcentajes": [
                datos_cluster["porcentaje"]
                for datos_cluster in distribucion_clusters.values()
            ],
            "tipos": [
                datos_cluster["tipo"]
                for datos_cluster in distribucion_clusters.values()
            ],
        },
        "evaluacion_eps": {
            "eps": [resultado["eps"] for resultado in evaluacion_eps],
            "cantidad_clusters": [
                resultado["cantidad_clusters"] for resultado in evaluacion_eps
            ],
            "cantidad_ruido": [
                resultado["cantidad_ruido"] for resultado in evaluacion_eps
            ],
            "porcentaje_ruido": [
                resultado["porcentaje_ruido"] for resultado in evaluacion_eps
            ],
        },
        "metricas": {
            "nombres": [
                "Silhouette",
                "Calinski-Harabasz",
                "Davies-Bouldin",
            ],
            "valores": [
                metricas["silhouette_score"],
                metricas["calinski_harabasz_score"],
                metricas["davies_bouldin_score"],
            ],
        },
    }


def obtener_contexto_dbscan(
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict:
    """Funcion principal del flujo Dataset -> PCA -> DBSCAN."""
    preparacion = preparar_datos_dbscan()
    resultado_pca = aplicar_pca_para_dbscan(preparacion["datos_escalados"])
    datos_pca = resultado_pca["componentes_df"]
    resultado_dbscan = ejecutar_dbscan(
        datos_pca,
        eps=eps,
        min_samples=min_samples,
    )
    resumen_clusters = identificar_clusters_y_ruido(resultado_dbscan["etiquetas"])
    distribucion = obtener_distribucion_clusters_dbscan(resultado_dbscan["etiquetas"])
    metricas = calcular_metricas_dbscan(datos_pca, resultado_dbscan["etiquetas"])
    evaluacion_eps = evaluar_valores_eps(
        datos_pca,
        valores_eps=EPS_VALUES_TO_EVALUATE,
        min_samples=min_samples,
    )
    conclusiones = generar_conclusiones_dbscan(
        resumen_clusters,
        distribucion,
        eps,
        min_samples,
        evaluacion_eps,
    )
    chart_data = preparar_datos_visualizacion_dbscan(
        preparacion["datos_originales"],
        datos_pca,
        resultado_dbscan["etiquetas"],
        distribucion,
        evaluacion_eps,
        metricas,
    )

    return {
        "parametros": {
            "eps": float(eps),
            "min_samples": int(min_samples),
        },
        "variables_utilizadas": preparacion["variables_utilizadas"],
        "cantidad_registros": preparacion["cantidad_registros"],
        "evidencia_escalado": preparacion["evidencia_escalado"],
        "componentes_pca": resultado_pca["nombres_componentes"],
        "varianza_explicada_pca": resultado_pca["varianza_explicada_porcentaje"],
        "varianza_acumulada_pca": resultado_pca["varianza_acumulada_porcentaje"],
        "clusters_encontrados": resumen_clusters["cantidad_clusters"],
        "ruido_detectado": resumen_clusters["cantidad_ruido"],
        "porcentaje_ruido": resumen_clusters["porcentaje_ruido"],
        "distribucion_clusters": distribucion,
        "metricas": metricas,
        "evaluacion_eps": evaluacion_eps,
        "chart_data": chart_data,
        "conclusiones": conclusiones,
    }
