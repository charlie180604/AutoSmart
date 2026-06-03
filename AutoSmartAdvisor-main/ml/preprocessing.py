"""Preprocesamiento basico de datos para AutoSmart Advisor."""

from pathlib import Path
import unicodedata

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_CSV = RAW_DIR / "cardekho_vehicle_dataset.csv"
RAW_CSV_FALLBACK = RAW_DIR / "CAR DETAILS FROM CAR DEKHO.csv"
OUTPUT_CSV = PROCESSED_DIR / "autos_limpios.csv"


def obtener_ruta_dataset():
    if RAW_CSV.exists():
        return RAW_CSV
    if RAW_CSV_FALLBACK.exists():
        return RAW_CSV_FALLBACK
    raise FileNotFoundError(
        "No se encontro el dataset de vehiculos en data/raw. "
        "Se esperaba cardekho_vehicle_dataset.csv o CAR DETAILS FROM CAR DEKHO.csv."
    )


def normalizar_texto(valor):
    if valor is None:
        return ""
    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(caracter for caracter in texto if not unicodedata.combining(caracter))


def normalizar_combustible(valor):
    texto = normalizar_texto(valor)
    mapa = {
        "petrol": "gasolina",
        "diesel": "diesel",
        "cng": "gas",
        "lpg": "gas",
        "electric": "electrico",
        "electricidad": "electrico",
        "hybrid": "hibrido",
    }
    return mapa.get(texto, texto)


def normalizar_transmision(valor):
    texto = normalizar_texto(valor)
    mapa = {
        "manual": "manual",
        "automatic": "automatica",
        "automatica": "automatica",
    }
    return mapa.get(texto, texto)


def preparar_autos():
    ruta_dataset = obtener_ruta_dataset()
    autos = pd.read_csv(ruta_dataset)

    columnas = {
        "name": "nombre",
        "year": "anio",
        "selling_price": "precio",
        "km_driven": "kilometraje",
        "fuel": "combustible",
        "seller_type": "tipo_vendedor",
        "transmission": "transmision",
        "owner": "propietarios",
    }

    faltantes = [columna for columna in columnas if columna not in autos.columns]
    if faltantes:
        raise ValueError(
            "El dataset no contiene las columnas esperadas: "
            + ", ".join(faltantes)
        )

    autos = autos.rename(columns=columnas)
    autos = autos[list(columnas.values())].copy()

    for columna in ("nombre", "combustible", "transmision", "tipo_vendedor", "propietarios"):
        autos[columna] = autos[columna].astype(str).str.strip()

    autos["anio"] = pd.to_numeric(autos["anio"], errors="coerce")
    autos["precio"] = pd.to_numeric(autos["precio"], errors="coerce")
    autos["kilometraje"] = pd.to_numeric(autos["kilometraje"], errors="coerce")

    columnas_esenciales = ["nombre", "anio", "precio", "combustible", "transmision"]
    autos = autos.dropna(subset=columnas_esenciales)
    autos = autos[autos["nombre"].str.len() > 0]
    autos = autos[autos["combustible"].str.len() > 0]
    autos = autos[autos["transmision"].str.len() > 0]
    autos = autos[autos["precio"] > 0]

    autos["anio"] = autos["anio"].astype(int)
    autos["precio"] = autos["precio"].astype(int)
    autos["kilometraje"] = autos["kilometraje"].fillna(0).astype(int)
    autos["combustible"] = autos["combustible"].apply(normalizar_combustible)
    autos["transmision"] = autos["transmision"].apply(normalizar_transmision)
    autos["tipo_vendedor"] = autos["tipo_vendedor"].apply(normalizar_texto)
    autos["propietarios"] = autos["propietarios"].apply(normalizar_texto)

    autos["tipo"] = "Auto"
    autos["rendimiento"] = "No especificado"
    autos["pasajeros"] = 5
    autos["etiqueta"] = "Recomendado"

    autos = autos.drop_duplicates(subset=["nombre", "anio", "precio", "kilometraje"])
    autos = autos.sort_values(["precio", "anio"], ascending=[True, False])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    autos.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    return OUTPUT_CSV


if __name__ == "__main__":
    salida = preparar_autos()
    print(f"Archivo procesado generado: {salida}")
