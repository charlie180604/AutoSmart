"""Recomendaciones iniciales basadas en el dataset limpio."""

from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

try:
    from ml.preprocessing import OUTPUT_CSV, preparar_autos
except ModuleNotFoundError:
    from preprocessing import OUTPUT_CSV, preparar_autos


BASE_DIR = Path(__file__).resolve().parents[1]


def _normalizar_texto(valor):
    texto = str(valor or "").strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(caracter for caracter in texto if not unicodedata.combining(caracter))


def _capitalizar(valor):
    texto = str(valor or "").strip()
    if texto == "automatica":
        return "Automática"
    if texto == "electrico":
        return "Eléctrico"
    if texto == "hibrido":
        return "Híbrido"
    return texto.capitalize()


def _asegurar_dataset_procesado():
    if not OUTPUT_CSV.exists():
        preparar_autos()
    if not OUTPUT_CSV.exists():
        raise FileNotFoundError(
            "No se pudo generar data/processed/autos_limpios.csv."
        )


def _seleccionar_candidatos(autos, presupuesto, cantidad):
    if presupuesto <= 0:
        return autos.copy()

    rango_ideal = autos[
        (autos["precio"] >= presupuesto * 0.6)
        & (autos["precio"] <= presupuesto)
    ].copy()
    dentro_presupuesto = autos[autos["precio"] <= presupuesto].copy()
    muy_baratos = autos[autos["precio"] < presupuesto * 0.6].copy()

    cercanos = autos.copy()
    cercanos["distancia_presupuesto"] = (cercanos["precio"] - presupuesto).abs()
    cercanos = cercanos.sort_values("distancia_presupuesto").head(max(25, cantidad * 8))

    candidatos = pd.concat(
        [
            rango_ideal,
            cercanos,
            dentro_presupuesto.sort_values("precio", ascending=False).head(max(25, cantidad * 8)),
            muy_baratos.sort_values("precio", ascending=False).head(max(12, cantidad * 4)),
        ],
        ignore_index=True,
    )
    return candidatos.drop_duplicates(subset=["nombre", "anio", "precio", "kilometraje"])


def _calcular_compatibilidad(autos, datos_usuario):
    presupuesto = float(datos_usuario.get("presupuesto") or 0)
    combustible = _normalizar_texto(datos_usuario.get("combustible"))
    transmision = _normalizar_texto(datos_usuario.get("transmision"))
    prioridad = _normalizar_texto(datos_usuario.get("prioridad"))

    autos = autos.copy()
    autos["compatibilidad"] = 35.0

    if presupuesto > 0:
        proporcion_precio = autos["precio"] / presupuesto

        rango_ideal = proporcion_precio.between(0.6, 1.0)
        cercania_ideal = 1 - ((proporcion_precio - 0.85).abs() / 0.4).clip(lower=0, upper=1)
        autos["compatibilidad"] += cercania_ideal * 22

        autos.loc[rango_ideal, "compatibilidad"] += 4
        autos.loc[proporcion_precio < 0.6, "compatibilidad"] -= (0.6 - proporcion_precio) * 35
        autos.loc[proporcion_precio < 0.35, "compatibilidad"] -= 8
        autos.loc[proporcion_precio > 1.0, "compatibilidad"] -= ((proporcion_precio - 1.0) * 40).clip(upper=20)

    if combustible and combustible != "indiferente":
        autos.loc[autos["combustible"] == combustible, "compatibilidad"] += 12
        autos.loc[autos["combustible"] != combustible, "compatibilidad"] -= 4

    if transmision and transmision != "indiferente":
        autos.loc[autos["transmision"] == transmision, "compatibilidad"] += 12
        autos.loc[autos["transmision"] != transmision, "compatibilidad"] -= 4

    if prioridad == "ahorro":
        precio_mediano = autos["precio"].median()
        kilometraje_mediano = autos["kilometraje"].median()
        autos.loc[autos["precio"] <= precio_mediano, "compatibilidad"] += 4
        autos.loc[autos["kilometraje"] <= kilometraje_mediano, "compatibilidad"] += 3
        autos.loc[autos["combustible"].isin(["gasolina", "electrico", "hibrido"]), "compatibilidad"] += 2
    elif prioridad in ("comodidad", "seguridad", "lujo"):
        anio_mediano = autos["anio"].median()
        autos.loc[autos["anio"] >= anio_mediano, "compatibilidad"] += 6
        autos.loc[autos["kilometraje"] <= autos["kilometraje"].median(), "compatibilidad"] += 3
    elif prioridad == "potencia":
        autos.loc[autos["combustible"].isin(["diesel", "gasolina"]), "compatibilidad"] += 4

    autos["compatibilidad"] = np.clip(autos["compatibilidad"].round(), 0, 98).astype(int)
    return autos


def generar_recomendacion(datos_usuario, cantidad=3):
    """
    Genera recomendaciones iniciales desde data/processed/autos_limpios.csv.

    Esta funcion todavia no usa modelos de Machine Learning; solo aplica reglas
    simples de presupuesto, combustible y transmision sobre el dataset real.
    """
    _asegurar_dataset_procesado()
    autos = pd.read_csv(OUTPUT_CSV)

    if autos.empty:
        return []

    autos["combustible"] = autos["combustible"].apply(_normalizar_texto)
    autos["transmision"] = autos["transmision"].apply(_normalizar_texto)

    presupuesto = float(datos_usuario.get("presupuesto") or 0)
    candidatos = _seleccionar_candidatos(autos, presupuesto, cantidad)
    candidatos = _calcular_compatibilidad(candidatos, datos_usuario)
    candidatos = candidatos.sort_values(
        ["compatibilidad", "anio", "kilometraje", "precio"],
        ascending=[False, False, True, False],
    ).head(cantidad)

    recomendaciones = []
    for _, auto in candidatos.iterrows():
        recomendaciones.append(
            {
                "nombre": auto.get("nombre", "Auto sin nombre"),
                "tipo": auto.get("tipo", "Auto"),
                "combustible": _capitalizar(auto.get("combustible", "")),
                "transmision": _capitalizar(auto.get("transmision", "")),
                "precio": int(auto.get("precio", 0)),
                "anio": int(auto.get("anio", 0)),
                "rendimiento": auto.get("rendimiento", "No especificado"),
                "pasajeros": int(auto.get("pasajeros", 5)),
                "compatibilidad": int(auto.get("compatibilidad", 0)),
                "etiqueta": auto.get("etiqueta", "Recomendado"),
            }
        )

    if recomendaciones:
        recomendaciones[0]["etiqueta"] = "Mejor opción"
    if len(recomendaciones) > 1:
        recomendaciones[1]["etiqueta"] = "Opción recomendada"
    if len(recomendaciones) > 2:
        recomendaciones[2]["etiqueta"] = "Alternativa viable"

    return recomendaciones
