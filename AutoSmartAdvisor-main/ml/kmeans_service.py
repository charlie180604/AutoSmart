"""Servicio backend para clustering con K-Means.

Esta fase no modifica la interfaz ni el recomendador. El modulo prepara una
base reutilizable para segmentar vehiculos despues de escalar variables y
reducir dimensionalidad con PCA.
"""

import pandas as pd
from sklearn.cluster import KMeans
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


DEFAULT_N_CLUSTERS = 3
DEFAULT_RANDOM_STATE = 42
MIN_K = 2
MAX_K = 10


def _filtrar_variables_utiles(datos: pd.DataFrame) -> pd.DataFrame:
    """Elimina variables constantes o sin variabilidad util para clustering."""
    columnas_utiles = [
        columna for columna in datos.columns if datos[columna].nunique(dropna=True) > 1
    ]
    return datos[columnas_utiles].copy()


def preparar_datos_kmeans(autos=None) -> dict:
    """Carga, limpia y escala las variables numericas utiles para K-Means."""
    if autos is None:
        autos = cargar_dataset()

    datos_numericos = obtener_variables_numericas_limpias(
        autos,
        columnas=VARIABLES_CANDIDATAS_PCA,
        eliminar_nulos=True,
    )
    datos_utiles = _filtrar_variables_utiles(datos_numericos)

    if datos_utiles.empty:
        raise ValueError("No hay variables numericas utiles para aplicar K-Means.")

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


def aplicar_pca_para_kmeans(datos_escalados: pd.DataFrame) -> dict:
    """Aplica el PCA ya implementado y devuelve sus componentes."""
    resultado_pca = aplicar_pca(datos_escalados)
    componentes = resultado_pca["componentes"]
    nombres_componentes = resultado_pca["nombres_componentes"]

    componentes_df = pd.DataFrame(
        componentes,
        columns=nombres_componentes,
        index=datos_escalados.index,
    )

    return {
        **resultado_pca,
        "componentes_df": componentes_df,
    }


def ejecutar_kmeans(
    datos_pca: pd.DataFrame,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Ejecuta K-Means sobre los componentes PCA."""
    if n_clusters < 2:
        raise ValueError("K-Means requiere al menos 2 clusters.")
    if n_clusters >= len(datos_pca):
        raise ValueError("El numero de clusters debe ser menor que los registros.")

    modelo = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    etiquetas = modelo.fit_predict(datos_pca)

    return {
        "modelo": modelo,
        "etiquetas": etiquetas,
        "centroides": modelo.cluster_centers_.round(4).tolist(),
        "centroides_array": modelo.cluster_centers_,
        "inercia": round(float(modelo.inertia_), 4),
        "n_clusters": int(n_clusters),
    }


def evaluar_rango_k(
    datos_pca: pd.DataFrame,
    min_k: int = MIN_K,
    max_k: int = MAX_K,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> list[dict]:
    """Evalua multiples valores de K para el metodo del codo y Silhouette."""
    maximo_permitido = min(max_k, len(datos_pca) - 1)
    resultados = []

    for k in range(min_k, maximo_permitido + 1):
        resultado = ejecutar_kmeans(
            datos_pca,
            n_clusters=k,
            random_state=random_state,
        )
        metricas = calcular_metricas_kmeans(datos_pca, resultado["etiquetas"])
        resultados.append(
            {
                "k": int(k),
                "inercia": resultado["inercia"],
                "silhouette_score": metricas["silhouette_score"],
            }
        )

    return resultados


def recomendar_mejor_k(evaluacion_k: list[dict]) -> dict:
    """Selecciona el mejor K usando Silhouette como criterio principal."""
    evaluaciones_validas = [
        evaluacion
        for evaluacion in evaluacion_k
        if evaluacion["silhouette_score"] is not None
    ]
    if not evaluaciones_validas:
        return {
            "k_recomendado": DEFAULT_N_CLUSTERS,
            "criterio": "Valor por defecto por falta de metricas comparables.",
        }

    mejor = max(
        evaluaciones_validas,
        key=lambda evaluacion: (
            evaluacion["silhouette_score"],
            -evaluacion["k"],
        ),
    )
    return {
        "k_recomendado": int(mejor["k"]),
        "silhouette_score": mejor["silhouette_score"],
        "criterio": "Mayor Silhouette Score entre K=2 y K=10.",
    }


def calcular_metricas_kmeans(datos_pca: pd.DataFrame, etiquetas) -> dict:
    """Calcula metricas de evaluacion para los clusters generados."""
    cantidad_clusters = len(set(etiquetas))
    if cantidad_clusters < 2:
        return {
            "silhouette_score": None,
            "calinski_harabasz_score": None,
            "davies_bouldin_score": None,
        }

    return {
        "silhouette_score": round(float(silhouette_score(datos_pca, etiquetas)), 4),
        "calinski_harabasz_score": round(
            float(calinski_harabasz_score(datos_pca, etiquetas)),
            4,
        ),
        "davies_bouldin_score": round(
            float(davies_bouldin_score(datos_pca, etiquetas)),
            4,
        ),
    }


def obtener_distribucion_clusters(etiquetas) -> dict:
    """Devuelve la cantidad de registros asignados a cada cluster."""
    serie = pd.Series(etiquetas, name="cluster")
    distribucion = serie.value_counts().sort_index()
    total = int(distribucion.sum())

    return {
        int(cluster): {
            "cantidad": int(cantidad),
            "porcentaje": round((int(cantidad) / total) * 100, 2),
        }
        for cluster, cantidad in distribucion.items()
    }


def generar_conclusiones_kmeans(
    metricas: dict,
    distribucion_clusters: dict,
    n_clusters: int,
    recomendacion_k: dict | None = None,
) -> list[str]:
    """Genera conclusiones automaticas a partir de metricas y distribucion."""
    conclusiones = [
        f"K-Means segmentó el dataset en {n_clusters} clusters.",
    ]

    silhouette = metricas.get("silhouette_score")
    if silhouette is None:
        conclusiones.append(
            "No fue posible calcular Silhouette porque no hay suficientes clusters."
        )
    elif silhouette >= 0.5:
        conclusiones.append(
            "La separacion entre clusters es clara segun el indice Silhouette."
        )
    elif silhouette >= 0.25:
        conclusiones.append(
            "La separacion entre clusters es moderada y puede ser util para exploracion."
        )
    else:
        conclusiones.append(
            "La separacion entre clusters es baja; conviene comparar otros valores de k."
        )

    porcentajes = [
        datos_cluster["porcentaje"]
        for datos_cluster in distribucion_clusters.values()
    ]
    if porcentajes:
        diferencia = max(porcentajes) - min(porcentajes)
        cluster_mas_grande = max(
            distribucion_clusters,
            key=lambda cluster: distribucion_clusters[cluster]["cantidad"],
        )
        cluster_mas_pequeno = min(
            distribucion_clusters,
            key=lambda cluster: distribucion_clusters[cluster]["cantidad"],
        )
        conclusiones.append(
            "El cluster mas grande es el "
            f"{cluster_mas_grande} con "
            f"{distribucion_clusters[cluster_mas_grande]['cantidad']} registros."
        )
        conclusiones.append(
            "El cluster mas pequeno es el "
            f"{cluster_mas_pequeno} con "
            f"{distribucion_clusters[cluster_mas_pequeno]['cantidad']} registros."
        )

        if diferencia <= 20:
            conclusiones.append(
                "La distribucion de vehiculos entre clusters es relativamente equilibrada."
            )
        else:
            conclusiones.append(
                "La distribucion de vehiculos entre clusters muestra segmentos de distinto tamano."
            )

    if recomendacion_k:
        conclusiones.append(
            "El analisis comparativo recomienda usar "
            f"K={recomendacion_k['k_recomendado']} segun el criterio: "
            f"{recomendacion_k['criterio']}"
        )

    return conclusiones


def preparar_datos_visualizacion_kmeans(
    datos_originales: pd.DataFrame,
    datos_pca: pd.DataFrame,
    etiquetas,
    resultado_kmeans: dict,
    distribucion_clusters: dict,
    evaluacion_k: list[dict],
) -> dict:
    """Prepara estructuras listas para graficas Plotly de K-Means."""
    datos_visuales = pd.DataFrame(
        {
            "PC1": datos_pca["PC1"].round(4),
            "PC2": datos_pca["PC2"].round(4),
            "cluster": [int(etiqueta) for etiqueta in etiquetas],
        },
        index=datos_pca.index,
    )

    for columna in datos_originales.columns:
        datos_visuales[columna] = datos_originales[columna].round(2)

    centroides = []
    for cluster, centroide in enumerate(resultado_kmeans["centroides_array"]):
        datos_cluster = distribucion_clusters.get(cluster, {})
        centroides.append(
            {
                "cluster": int(cluster),
                "PC1": round(float(centroide[0]), 4),
                "PC2": round(float(centroide[1]), 4),
                "cantidad_registros": int(datos_cluster.get("cantidad", 0)),
                "porcentaje": float(datos_cluster.get("porcentaje", 0)),
            }
        )

    return {
        "metodo_codo": {
            "k": [resultado["k"] for resultado in evaluacion_k],
            "inercia": [resultado["inercia"] for resultado in evaluacion_k],
            "silhouette_score": [
                resultado["silhouette_score"] for resultado in evaluacion_k
            ],
        },
        "distribucion_clusters": {
            "clusters": [
                int(cluster) for cluster in distribucion_clusters.keys()
            ],
            "cantidades": [
                datos_cluster["cantidad"]
                for datos_cluster in distribucion_clusters.values()
            ],
            "porcentajes": [
                datos_cluster["porcentaje"]
                for datos_cluster in distribucion_clusters.values()
            ],
        },
        "pca_2d_clusters": datos_visuales.to_dict(orient="records"),
        "centroides": centroides,
    }


def generar_perfiles_clusters(
    datos_originales: pd.DataFrame,
    etiquetas,
) -> list[dict]:
    """Calcula perfiles descriptivos por cluster usando datos reales."""
    datos = datos_originales.copy()
    datos["cluster"] = [int(etiqueta) for etiqueta in etiquetas]

    perfiles = []
    for cluster, grupo in datos.groupby("cluster"):
        perfiles.append(
            {
                "cluster": int(cluster),
                "cantidad_registros": int(len(grupo)),
                "precio_promedio": round(float(grupo["precio"].mean()), 2),
                "anio_promedio": round(float(grupo["anio"].mean()), 1),
                "kilometraje_promedio": round(
                    float(grupo["kilometraje"].mean()),
                    2,
                ),
            }
        )

    return perfiles


def generar_conclusiones_perfiles(
    perfiles_clusters: list[dict],
    datos_originales: pd.DataFrame,
) -> list[str]:
    """Genera conclusiones automaticas sobre el perfil de cada cluster."""
    if not perfiles_clusters:
        return []

    precio_general = datos_originales["precio"].mean()
    kilometraje_general = datos_originales["kilometraje"].mean()
    anio_general = datos_originales["anio"].mean()
    cluster_mayor = max(perfiles_clusters, key=lambda perfil: perfil["cantidad_registros"])
    cluster_mayor_km = max(
        perfiles_clusters,
        key=lambda perfil: perfil["kilometraje_promedio"],
    )

    conclusiones = [
        "El cluster "
        f"{cluster_mayor['cluster']} representa la mayor parte del dataset "
        f"con {cluster_mayor['cantidad_registros']} vehiculos.",
        "El cluster "
        f"{cluster_mayor_km['cluster']} contiene los vehiculos con mayor "
        "kilometraje promedio.",
    ]

    for perfil in perfiles_clusters:
        comparacion_precio = (
            "superiores"
            if perfil["precio_promedio"] > precio_general
            else "inferiores"
        )
        comparacion_anio = (
            "mas recientes"
            if perfil["anio_promedio"] > anio_general
            else "mas antiguos"
        )
        comparacion_km = (
            "mayor kilometraje"
            if perfil["kilometraje_promedio"] > kilometraje_general
            else "menor kilometraje"
        )
        conclusiones.append(
            "Cluster "
            f"{perfil['cluster']} agrupa vehiculos con precios "
            f"{comparacion_precio} al promedio, unidades {comparacion_anio} "
            f"y {comparacion_km}."
        )

    return conclusiones


def generar_interpretaciones_segmentos(
    perfiles_clusters: list[dict],
) -> list[dict]:
    """Describe automaticamente que representa cada cluster comparando perfiles."""
    if not perfiles_clusters:
        return []

    precios = [perfil["precio_promedio"] for perfil in perfiles_clusters]
    anios = [perfil["anio_promedio"] for perfil in perfiles_clusters]
    kilometrajes = [
        perfil["kilometraje_promedio"] for perfil in perfiles_clusters
    ]
    cantidades = [perfil["cantidad_registros"] for perfil in perfiles_clusters]

    precio_promedio_general = sum(precios) / len(precios)
    anio_promedio_general = sum(anios) / len(anios)
    kilometraje_promedio_general = sum(kilometrajes) / len(kilometrajes)
    cantidad_promedio_general = sum(cantidades) / len(cantidades)

    precio_maximo = max(precios)
    precio_minimo = min(precios)
    anio_maximo = max(anios)
    anio_minimo = min(anios)
    kilometraje_maximo = max(kilometrajes)
    kilometraje_minimo = min(kilometrajes)
    cantidad_maxima = max(cantidades)

    interpretaciones = []
    for perfil in perfiles_clusters:
        rasgos = []

        if perfil["precio_promedio"] == precio_maximo:
            rasgos.append("precios mas altos del conjunto")
        elif perfil["precio_promedio"] == precio_minimo:
            rasgos.append("precios mas economicos")
        elif perfil["precio_promedio"] > precio_promedio_general:
            rasgos.append("precios superiores al promedio")
        else:
            rasgos.append("precios intermedios o accesibles")

        if perfil["anio_promedio"] == anio_maximo:
            rasgos.append("vehiculos mas recientes")
        elif perfil["anio_promedio"] == anio_minimo:
            rasgos.append("vehiculos de mayor antiguedad")
        elif perfil["anio_promedio"] > anio_promedio_general:
            rasgos.append("modelos relativamente modernos")
        else:
            rasgos.append("modelos con mayor antiguedad relativa")

        if perfil["kilometraje_promedio"] == kilometraje_minimo:
            rasgos.append("menor kilometraje promedio")
        elif perfil["kilometraje_promedio"] == kilometraje_maximo:
            rasgos.append("mayor kilometraje promedio")
        elif perfil["kilometraje_promedio"] > kilometraje_promedio_general:
            rasgos.append("uso acumulado superior al promedio")
        else:
            rasgos.append("uso acumulado moderado")

        if perfil["cantidad_registros"] == cantidad_maxima:
            tamano = "representa el segmento mas grande del dataset"
        elif perfil["cantidad_registros"] >= cantidad_promedio_general:
            tamano = "tiene una presencia relevante dentro del dataset"
        else:
            tamano = "corresponde a un segmento mas especifico del mercado"

        descripcion = (
            f"Cluster {perfil['cluster']} agrupa vehiculos con "
            + ", ".join(rasgos)
            + f"; ademas, {tamano}."
        )
        interpretaciones.append(
            {
                "cluster": perfil["cluster"],
                "descripcion": descripcion,
                "precio_promedio": perfil["precio_promedio"],
                "anio_promedio": perfil["anio_promedio"],
                "kilometraje_promedio": perfil["kilometraje_promedio"],
                "cantidad_registros": perfil["cantidad_registros"],
            }
        )

    return interpretaciones


def obtener_contexto_kmeans(
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Funcion principal que ejecuta el flujo Dataset -> PCA -> K-Means."""
    preparacion = preparar_datos_kmeans()
    resultado_pca = aplicar_pca_para_kmeans(preparacion["datos_escalados"])
    datos_pca = resultado_pca["componentes_df"]
    resultado_kmeans = ejecutar_kmeans(
        datos_pca,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    evaluacion_k = evaluar_rango_k(
        datos_pca,
        min_k=MIN_K,
        max_k=MAX_K,
        random_state=random_state,
    )
    recomendacion_k = recomendar_mejor_k(evaluacion_k)
    metricas = calcular_metricas_kmeans(
        datos_pca,
        resultado_kmeans["etiquetas"],
    )
    distribucion = obtener_distribucion_clusters(resultado_kmeans["etiquetas"])
    perfiles_clusters = generar_perfiles_clusters(
        preparacion["datos_originales"],
        resultado_kmeans["etiquetas"],
    )
    conclusiones_perfiles = generar_conclusiones_perfiles(
        perfiles_clusters,
        preparacion["datos_originales"],
    )
    interpretaciones_segmentos = generar_interpretaciones_segmentos(
        perfiles_clusters,
    )
    conclusiones = generar_conclusiones_kmeans(
        metricas,
        distribucion,
        n_clusters,
        recomendacion_k,
    )
    chart_data = preparar_datos_visualizacion_kmeans(
        preparacion["datos_originales"],
        datos_pca,
        resultado_kmeans["etiquetas"],
        resultado_kmeans,
        distribucion,
        evaluacion_k,
    )

    return {
        "variables_utilizadas": preparacion["variables_utilizadas"],
        "cantidad_registros": preparacion["cantidad_registros"],
        "evidencia_escalado": preparacion["evidencia_escalado"],
        "componentes_pca": resultado_pca["nombres_componentes"],
        "varianza_explicada_pca": resultado_pca["varianza_explicada_porcentaje"],
        "varianza_acumulada_pca": resultado_pca["varianza_acumulada_porcentaje"],
        "n_clusters": resultado_kmeans["n_clusters"],
        "k_recomendado": recomendacion_k,
        "evaluacion_k": evaluacion_k,
        "inercia": resultado_kmeans["inercia"],
        "centroides": resultado_kmeans["centroides"],
        "metricas": metricas,
        "distribucion_clusters": distribucion,
        "perfiles_clusters": perfiles_clusters,
        "conclusiones_perfiles": conclusiones_perfiles,
        "interpretaciones_segmentos": interpretaciones_segmentos,
        "chart_data": chart_data,
        "conclusiones": conclusiones + conclusiones_perfiles,
    }
