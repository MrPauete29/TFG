from flask import Flask, request, render_template, redirect, url_for
from riesgos import evaluar_riesgo_corazon, evaluar_riesgo_ictus

app = Flask(__name__)


@app.route("/")
def index():
    return redirect(url_for("sign_in"))

@app.route("/inicio")
def sign_in():
    return render_template("sign_in.html")


@app.route("/menu")
def menu_principal():
    return render_template("menu-principal.html")


@app.route("/corazon")
def formulario_corazon():
    return render_template("corazon.html")

@app.route("/ictus")
def formulario_ictus():
    return render_template("ictus.html")


@app.route("/procesar_corazon", methods=["POST"])
def procesar_corazon():
    datos = request.form.to_dict()
    resultado = evaluar_riesgo_corazon(datos)
    return render_template("resultados.html", resultado=resultado, volver_a="/corazon")

@app.route("/procesar_ictus", methods=["POST"])
def procesar_ictus():
    datos = request.form.to_dict()
    resultado = evaluar_riesgo_ictus(datos)
    return render_template("resultados.html", resultado=resultado, volver_a="/ictus")

if __name__ == "__main__":
    app.run(debug=True)
