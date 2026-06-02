"""Graficas academicas de analisis exploratorio con Matplotlib.

Este modulo genera evidencia visual estatica para cumplir la rubrica del
proyecto. No modifica la interfaz ni reemplaza las graficas interactivas
existentes en /analisis.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.exploratory_analysis import (
    calcular_matriz_correlacion,
    cargar_dataset,
    obtener_variables_numericas_limpias,
)


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "static" / "analysis"

COLUMNAS_GRAFICAS = ("precio", "anio", "kilometraje")
ARCHIVOS_GRAFICAS = {
    "histograma_precio": "histograma_precio.png",
    "boxplot_precio": "boxplot_precio.png",
    "scatter_precio_anio": "scatter_precio_anio.png",
    "scatter_precio_kilometraje": "scatter_precio_kilometraje.png",
    "heatmap_correlacion": "heatmap_correlacion.png",
}


def _preparar_directorio_salida(output_dir: str | Path = OUTPUT_DIR) -> Path:
    """Crea la carpeta de salida si no existe."""
    ruta = Path(output_dir)
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def _obtener_datos_numericos(autos: pd.DataFrame) -> pd.DataFrame:
    """Obtiene las variables numericas requeridas para las graficas."""
    datos = obtener_variables_numericas_limpias(
        autos,
        columnas=COLUMNAS_GRAFICAS,
        eliminar_nulos=True,
    )
    columnas_faltantes = [
        columna for columna in COLUMNAS_GRAFICAS if columna not in datos.columns
    ]
    if columnas_faltantes:
        raise ValueError(
            "No se encontraron columnas numericas necesarias: "
            + ", ".join(columnas_faltantes)
        )
    return datos


def _guardar_figura(figura, ruta_archivo: Path) -> str:
    """Guarda una figura en disco y cierra el recurso de Matplotlib."""
    try:
        figura.tight_layout()
        figura.savefig(ruta_archivo, dpi=140, bbox_inches="tight")
        return str(ruta_archivo)
    finally:
        plt.close(figura)


def generar_histograma_precios(
    datos: pd.DataFrame,
    output_dir: str | Path = OUTPUT_DIR,
) -> str:
    """Genera el histograma de precios."""
    ruta = Path(output_dir) / ARCHIVOS_GRAFICAS["histograma_precio"]
    figura, eje = plt.subplots(figsize=(9, 5))
    eje.hist(datos["precio"], bins=30, color="#38bdf8", edgecolor="#2563eb")
    eje.set_title("Distribución de precios")
    eje.set_xlabel("Precio")
    eje.set_ylabel("Cantidad de vehículos")
    eje.grid(axis="y", alpha=0.25)
    return _guardar_figura(figura, ruta)


def generar_boxplot_precios(
    datos: pd.DataFrame,
    output_dir: str | Path = OUTPUT_DIR,
) -> str:
    """Genera el boxplot de precios."""
    ruta = Path(output_dir) / ARCHIVOS_GRAFICAS["boxplot_precio"]
    figura, eje = plt.subplots(figsize=(7, 5))
    eje.boxplot(
        datos["precio"],
        vert=True,
        patch_artist=True,
        boxprops={"facecolor": "#dbeafe", "color": "#2563eb"},
        medianprops={"color": "#0f172a", "linewidth": 2},
        whiskerprops={"color": "#2563eb"},
        capprops={"color": "#2563eb"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#f59e0b",
            "markeredgecolor": "#f59e0b",
            "alpha": 0.45,
        },
    )
    eje.set_title("Boxplot de precios")
    eje.set_ylabel("Precio")
    eje.grid(axis="y", alpha=0.25)
    return _guardar_figura(figura, ruta)


def generar_scatter_precio_anio(
    datos: pd.DataFrame,
    output_dir: str | Path = OUTPUT_DIR,
) -> str:
    """Genera la grafica de dispersion Precio vs Año."""
    ruta = Path(output_dir) / ARCHIVOS_GRAFICAS["scatter_precio_anio"]
    figura, eje = plt.subplots(figsize=(9, 5))
    eje.scatter(datos["anio"], datos["precio"], alpha=0.45, color="#2563eb", s=18)
    eje.set_title("Precio vs Año")
    eje.set_xlabel("Año")
    eje.set_ylabel("Precio")
    eje.grid(alpha=0.25)
    return _guardar_figura(figura, ruta)


def generar_scatter_precio_kilometraje(
    datos: pd.DataFrame,
    output_dir: str | Path = OUTPUT_DIR,
) -> str:
    """Genera la grafica de dispersion Precio vs Kilometraje."""
    ruta = Path(output_dir) / ARCHIVOS_GRAFICAS["scatter_precio_kilometraje"]
    figura, eje = plt.subplots(figsize=(9, 5))
    eje.scatter(
        datos["kilometraje"],
        datos["precio"],
        alpha=0.42,
        color="#16a34a",
        s=18,
    )
    eje.set_title("Precio vs Kilometraje")
    eje.set_xlabel("Kilometraje")
    eje.set_ylabel("Precio")
    eje.grid(alpha=0.25)
    return _guardar_figura(figura, ruta)


def generar_heatmap_correlacion(
    autos: pd.DataFrame,
    output_dir: str | Path = OUTPUT_DIR,
) -> str:
    """Genera el heatmap de correlacion para precio, anio y kilometraje."""
    ruta = Path(output_dir) / ARCHIVOS_GRAFICAS["heatmap_correlacion"]
    correlacion = calcular_matriz_correlacion(
        autos,
        columnas=COLUMNAS_GRAFICAS,
    )
    matriz = pd.DataFrame(
        correlacion["matriz"],
        index=correlacion["columnas"],
        columns=correlacion["columnas"],
    )

    figura, eje = plt.subplots(figsize=(7, 5.6))
    imagen = eje.imshow(matriz.values, cmap="coolwarm", vmin=-1, vmax=1)
    eje.set_title("Matriz de correlación")
    eje.set_xticks(np.arange(len(matriz.columns)))
    eje.set_yticks(np.arange(len(matriz.index)))
    eje.set_xticklabels(matriz.columns)
    eje.set_yticklabels(matriz.index)

    for fila in range(len(matriz.index)):
        for columna in range(len(matriz.columns)):
            valor = matriz.iloc[fila, columna]
            eje.text(
                columna,
                fila,
                f"{valor:.2f}",
                ha="center",
                va="center",
                color="#0f172a",
                fontweight="bold",
            )

    figura.colorbar(imagen, ax=eje, fraction=0.046, pad=0.04)
    return _guardar_figura(figura, ruta)


def generar_graficas_matplotlib(output_dir: str | Path = OUTPUT_DIR) -> dict:
    """Genera las 5 graficas requeridas por la rubrica usando Matplotlib."""
    carpeta_salida = _preparar_directorio_salida(output_dir)
    autos = cargar_dataset()
    datos = _obtener_datos_numericos(autos)

    funciones_graficas = (
        lambda: generar_histograma_precios(datos, carpeta_salida),
        lambda: generar_boxplot_precios(datos, carpeta_salida),
        lambda: generar_scatter_precio_anio(datos, carpeta_salida),
        lambda: generar_scatter_precio_kilometraje(datos, carpeta_salida),
        lambda: generar_heatmap_correlacion(autos, carpeta_salida),
    )

    rutas_generadas = []
    errores = []
    try:
        for generar in funciones_graficas:
            try:
                rutas_generadas.append(generar())
            except (KeyError, ValueError, OSError, RuntimeError) as error:
                errores.append(str(error))
    finally:
        plt.close("all")

    return {
        "cantidad_graficas_generadas": len(rutas_generadas),
        "rutas_archivos": rutas_generadas,
        "errores": errores,
    }
