"""Analisis exploratorio reutilizable para AutoSmartAdvisor.

Este modulo no modifica el flujo de recomendacion. Solo prepara funciones
base para inspeccionar el dataset procesado y reutilizar sus variables
numericas en futuras etapas de Machine Learning.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATASET = BASE_DIR / "data" / "processed" / "autos_limpios.csv"

DESCRIPTIVE_COLUMNS = ("precio", "anio", "kilometraje", "rendimiento")
CORRELATION_COLUMNS = ("precio", "anio", "kilometraje", "rendimiento", "pasajeros")
REQUIRED_COLUMNS = (
    "nombre",
    "anio",
    "precio",
    "kilometraje",
    "combustible",
    "transmision",
)


def cargar_dataset(ruta_dataset: str | Path = PROCESSED_DATASET) -> pd.DataFrame:
    """Carga y valida el dataset procesado de autos.

    Args:
        ruta_dataset: Ruta al CSV procesado. Por defecto usa
            data/processed/autos_limpios.csv.

    Returns:
        DataFrame con el dataset cargado.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo esta vacio o no contiene columnas requeridas.
    """
    ruta = Path(ruta_dataset)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el dataset procesado: {ruta}")

    autos = pd.read_csv(ruta)
    validar_dataset(autos)
    return autos


def validar_dataset(autos: pd.DataFrame) -> dict:
    """Valida la estructura minima del dataset y devuelve un reporte simple."""
    if autos.empty:
        raise ValueError("El dataset procesado esta vacio.")

    columnas_faltantes = [
        columna for columna in REQUIRED_COLUMNS if columna not in autos.columns
    ]
    if columnas_faltantes:
        raise ValueError(
            "El dataset procesado no contiene las columnas requeridas: "
            + ", ".join(columnas_faltantes)
        )

    return {
        "registros": int(len(autos)),
        "columnas": list(autos.columns),
        "columnas_faltantes": columnas_faltantes,
        "valores_nulos": {
            columna: int(cantidad)
            for columna, cantidad in autos.isna().sum().to_dict().items()
        },
    }


def _columnas_existentes(autos: pd.DataFrame, columnas: Iterable[str]) -> list[str]:
    """Devuelve solo las columnas solicitadas que existen en el DataFrame."""
    return [columna for columna in columnas if columna in autos.columns]


def _serie_numerica_limpia(autos: pd.DataFrame, columna: str) -> pd.Series:
    """Convierte una columna a numerica y elimina valores no disponibles."""
    if columna not in autos.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(autos[columna], errors="coerce").dropna()


def calcular_estadisticas_descriptivas(
    autos: pd.DataFrame | None = None,
    columnas: Iterable[str] = DESCRIPTIVE_COLUMNS,
) -> dict:
    """Calcula estadisticas descriptivas para variables numericas relevantes.

    Las columnas no numericas o no disponibles se reportan con conteo 0 y
    metricas en None para que la salida sea facil de consumir desde una UI.
    """
    if autos is None:
        autos = cargar_dataset()

    estadisticas = {}
    for columna in columnas:
        serie = _serie_numerica_limpia(autos, columna)
        if serie.empty:
            estadisticas[columna] = {
                "cantidad_registros": 0,
                "promedio": None,
                "mediana": None,
                "minimo": None,
                "maximo": None,
                "desviacion_estandar": None,
            }
            continue

        estadisticas[columna] = {
            "cantidad_registros": int(serie.count()),
            "promedio": float(serie.mean()),
            "mediana": float(serie.median()),
            "minimo": float(serie.min()),
            "maximo": float(serie.max()),
            "desviacion_estandar": float(serie.std()),
        }

    return estadisticas


def detectar_outliers_iqr(autos: pd.DataFrame, columna: str) -> dict:
    """Detecta valores atipicos en una columna numerica usando el metodo IQR."""
    serie = _serie_numerica_limpia(autos, columna)
    if serie.empty:
        return {
            "columna": columna,
            "cantidad_outliers": 0,
            "valor_minimo_atipico": None,
            "valor_maximo_atipico": None,
            "limite_inferior": None,
            "limite_superior": None,
        }

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    outliers = serie[(serie < limite_inferior) | (serie > limite_superior)]

    return {
        "columna": columna,
        "cantidad_outliers": int(outliers.count()),
        "valor_minimo_atipico": None if outliers.empty else float(outliers.min()),
        "valor_maximo_atipico": None if outliers.empty else float(outliers.max()),
        "limite_inferior": float(limite_inferior),
        "limite_superior": float(limite_superior),
    }


def detectar_outliers_principales(autos: pd.DataFrame | None = None) -> dict:
    """Detecta outliers para precio y kilometraje."""
    if autos is None:
        autos = cargar_dataset()

    return {
        columna: detectar_outliers_iqr(autos, columna)
        for columna in ("precio", "kilometraje")
    }


def calcular_matriz_correlacion(
    autos: pd.DataFrame | None = None,
    columnas: Iterable[str] = CORRELATION_COLUMNS,
) -> dict:
    """Genera una matriz de correlacion para variables numericas disponibles."""
    if autos is None:
        autos = cargar_dataset()

    datos = obtener_variables_numericas_limpias(autos, columnas=columnas)
    if datos.empty:
        return {"columnas": [], "matriz": []}

    matriz = datos.corr().round(4)
    matriz = matriz.where(pd.notna(matriz), None)
    return {
        "columnas": list(matriz.columns),
        "matriz": matriz.to_dict(orient="records"),
    }


def obtener_variables_numericas_limpias(
    autos: pd.DataFrame | None = None,
    columnas: Iterable[str] = CORRELATION_COLUMNS,
    eliminar_nulos: bool = True,
) -> pd.DataFrame:
    """Obtiene variables numericas convertidas y listas para analisis/modelos."""
    if autos is None:
        autos = cargar_dataset()

    columnas_disponibles = _columnas_existentes(autos, columnas)
    datos = autos[columnas_disponibles].apply(pd.to_numeric, errors="coerce")

    if eliminar_nulos:
        datos = datos.dropna(axis=1, how="all").dropna()
    else:
        datos = datos.dropna(axis=1, how="all")

    return datos


def obtener_subconjunto_variables(
    autos: pd.DataFrame | None = None,
    columnas: Iterable[str] = CORRELATION_COLUMNS,
    solo_numericas: bool = True,
    eliminar_nulos: bool = True,
) -> pd.DataFrame:
    """Devuelve un subconjunto de columnas para analisis o modelado futuro."""
    if autos is None:
        autos = cargar_dataset()

    if solo_numericas:
        return obtener_variables_numericas_limpias(
            autos,
            columnas=columnas,
            eliminar_nulos=eliminar_nulos,
        )

    columnas_disponibles = _columnas_existentes(autos, columnas)
    datos = autos[columnas_disponibles].copy()
    return datos.dropna() if eliminar_nulos else datos


def normalizar_datos(
    datos: pd.DataFrame,
    metodo: str = "zscore",
) -> pd.DataFrame:
    """Normaliza datos numericos con z-score o min-max.

    Esta funcion queda lista para PCA, clustering y modelos supervisados.
    Las columnas constantes se convierten a 0 para evitar divisiones por cero.
    """
    if datos.empty:
        return datos.copy()

    datos_numericos = datos.apply(pd.to_numeric, errors="coerce")

    if metodo == "zscore":
        desviacion = datos_numericos.std().replace(0, pd.NA)
        normalizados = (datos_numericos - datos_numericos.mean()) / desviacion
    elif metodo == "minmax":
        rango = (datos_numericos.max() - datos_numericos.min()).replace(0, pd.NA)
        normalizados = (datos_numericos - datos_numericos.min()) / rango
    else:
        raise ValueError("Metodo de normalizacion no soportado. Usa 'zscore' o 'minmax'.")

    return normalizados.fillna(0)


def generar_resumen_exploratorio(autos: pd.DataFrame | None = None) -> dict:
    """Agrupa estadisticas, outliers y correlaciones en una salida reutilizable."""
    if autos is None:
        autos = cargar_dataset()

    return {
        "validacion": validar_dataset(autos),
        "estadisticas_descriptivas": calcular_estadisticas_descriptivas(autos),
        "outliers": detectar_outliers_principales(autos),
        "correlacion": calcular_matriz_correlacion(autos),
    }
