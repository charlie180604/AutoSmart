import unicodedata
from flask import Flask, render_template, request, redirect, url_for
from ml.analysis_service import obtener_contexto_analisis
from ml.classification_service import obtener_contexto_clasificacion
from ml.dbscan_service import obtener_contexto_dbscan
from ml.kmeans_service import obtener_contexto_kmeans
from ml.model_comparison_service import obtener_contexto_comparacion_modelos
from ml.predictor import generar_recomendacion
from ml.pca_service import obtener_contexto_pca
from ml.neural_regression_interpretation import enriquecer_contexto_red_neuronal
from ml.neural_regression_service import obtener_contexto_red_neuronal_regresion
from ml.regression_interpretation import enriquecer_contexto_regresion
from ml.regression_service import obtener_contexto_regresion
app = Flask(__name__)

def convertir_numero(valor):
    try:
        if valor is None or valor == "":
            return 0
        return float(valor)
    except (TypeError, ValueError):
        return 0


def normalizar_texto(valor):
    if valor is None:
        return ""
    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(caracter for caracter in texto if not unicodedata.combining(caracter))


def obtener_perfil(uso_principal):
    perfiles = {
        "ciudad": (
            "Usuario urbano económico",
            "Este perfil corresponde a usuarios que buscan un vehículo eficiente, accesible y adecuado para uso frecuente en ciudad.",
        ),
        "familiar": (
            "Usuario familiar",
            "Este perfil corresponde a usuarios que necesitan espacio, seguridad y comodidad para traslados con varios pasajeros.",
        ),
        "viajes": (
            "Usuario viajero",
            "Este perfil corresponde a usuarios que realizan trayectos largos y valoran comodidad, rendimiento y confiabilidad en carretera.",
        ),
        "trabajo": (
            "Usuario de trabajo",
            "Este perfil corresponde a usuarios que requieren un vehículo resistente, práctico y adecuado para actividades laborales.",
        ),
        "mixto": (
            "Usuario mixto",
            "Este perfil corresponde a usuarios que combinan ciudad, carretera y distintas necesidades de uso diario.",
        ),
    }
    return perfiles.get(
        uso_principal,
        (
            "Usuario general",
            "Este perfil corresponde a usuarios con necesidades generales de movilidad y preferencias variadas.",
        ),
    )


def evaluar_riesgo(tipo_compra, presupuesto, ingreso_mensual, gastos_mensuales, precio_auto):
    if precio_auto <= 0:
        return (
            "Sin evaluación",
            "No fue posible calcular el riesgo porque no se encontraron autos recomendados con precio válido.",
        )

    if tipo_compra == "contado":
        if presupuesto >= precio_auto:
            nivel = "Riesgo bajo"
        elif presupuesto >= precio_auto * 0.8:
            nivel = "Riesgo medio"
        else:
            nivel = "Riesgo alto"
    else:
        capacidad_pago = ingreso_mensual - gastos_mensuales
        mensualidad_simulada = precio_auto * 0.035

        if capacidad_pago <= 0:
            nivel = "Riesgo alto"
        elif capacidad_pago < mensualidad_simulada:
            nivel = "Riesgo medio"
        else:
            nivel = "Riesgo bajo"

    descripciones = {
        "Riesgo bajo": "La opción recomendada se encuentra dentro de un rango adecuado para el presupuesto y datos ingresados.",
        "Riesgo medio": "La compra podría ser viable, pero requiere revisar cuidadosamente la capacidad de pago.",
        "Riesgo alto": "La opción seleccionada podría representar una carga económica elevada para el usuario.",
    }

    return nivel, descripciones[nivel]


def formatear_precio(valor):
    return f"${valor:,.0f}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/formulario")
def formulario():
    return render_template("formulario.html")


@app.route("/resultados", methods=["GET", "POST"])
def resultados():
    if request.method == "GET":
        return redirect(url_for("formulario"))

    datos_usuario = {
        "presupuesto": convertir_numero(request.form.get("presupuesto")),
        "ingreso_mensual": convertir_numero(request.form.get("ingreso_mensual")),
        "tipo_compra": normalizar_texto(request.form.get("tipo_compra")),
        "gastos_mensuales": convertir_numero(request.form.get("gastos_mensuales")),
        "uso_principal": normalizar_texto(request.form.get("uso_principal")),
        "kilometros_semanales": convertir_numero(request.form.get("kilometros_semanales")),
        "numero_pasajeros": convertir_numero(request.form.get("numero_pasajeros")),
        "tipo_camino": normalizar_texto(request.form.get("tipo_camino")),
        "tipo_auto": normalizar_texto(request.form.get("tipo_auto")),
        "combustible": normalizar_texto(request.form.get("combustible")),
        "transmision": normalizar_texto(request.form.get("transmision")),
        "prioridad": normalizar_texto(request.form.get("prioridad")),
    }

    perfil_usuario, descripcion_perfil = obtener_perfil(datos_usuario["uso_principal"])
    autos_recomendados = generar_recomendacion(datos_usuario)

    precio_primer_auto = autos_recomendados[0]["precio"] if autos_recomendados else 0
    nivel_riesgo, descripcion_riesgo = evaluar_riesgo(
        datos_usuario["tipo_compra"],
        datos_usuario["presupuesto"],
        datos_usuario["ingreso_mensual"],
        datos_usuario["gastos_mensuales"],
        precio_primer_auto,
    )

    autos_formateados = []
    for auto in autos_recomendados:
        auto_formateado = auto.copy()
        auto_formateado["precio"] = formatear_precio(auto["precio"])
        autos_formateados.append(auto_formateado)

    return render_template(
        "resultados.html",
        perfil_usuario=perfil_usuario,
        descripcion_perfil=descripcion_perfil,
        nivel_riesgo=nivel_riesgo,
        descripcion_riesgo=descripcion_riesgo,
        autos_recomendados=autos_formateados,
    )


@app.route("/analisis")
def analisis():
    contexto_analisis = obtener_contexto_analisis()
    return render_template("analisis.html", **contexto_analisis)


@app.route("/pca")
def pca():
    contexto_pca = obtener_contexto_pca()
    return render_template("pca.html", **contexto_pca)


@app.route("/kmeans")
def kmeans():
    contexto_kmeans = obtener_contexto_kmeans()
    return render_template("kmeans.html", **contexto_kmeans)


@app.route("/dbscan")
def dbscan():
    contexto_dbscan = obtener_contexto_dbscan()
    return render_template("dbscan.html", **contexto_dbscan)

@app.route("/clasificacion")
def clasificacion():
    contexto_clasificacion = obtener_contexto_clasificacion()
    return render_template("clasificacion.html", **contexto_clasificacion)


@app.route("/regresion")
def regresion():
    contexto_regresion = enriquecer_contexto_regresion(obtener_contexto_regresion())
    return render_template("regresion.html", **contexto_regresion)


@app.route("/red-neuronal-regresion")
def red_neuronal_regresion():
    contexto_red_neuronal = enriquecer_contexto_red_neuronal(
        obtener_contexto_red_neuronal_regresion()
    )
    return render_template("red_neuronal_regresion.html", **contexto_red_neuronal)


@app.route("/comparacion-modelos")
def comparacion_modelos():
    contexto_comparacion = obtener_contexto_comparacion_modelos()
    return render_template("comparacion_modelos.html", **contexto_comparacion)


@app.route("/centro-visualizacion")
def centro_visualizacion():
    return render_template("centro_visualizacion_ml.html")


@app.route("/extras")
def extras():
    return render_template("extras.html")


@app.route("/acerca")
def acerca():
    return render_template("acerca.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
