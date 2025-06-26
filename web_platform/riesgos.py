def calcular_bmi(peso: float, altura: float) -> float:
    return peso / (altura ** 2)

def evaluar_riesgo_corazon(datos):
    riesgo = 0
    edad = int(datos["age"])
    bmi = calcular_bmi(float(datos["mass"]), float(datos["height"]))

    if 50 <= edad < 60:
        riesgo += 1
    elif edad >= 60:
        riesgo += 2


    if 20 <= bmi < 25:
        riesgo += 1
    elif bmi >= 25:
        riesgo += 2


    if datos["Diabetic"] == "1" or datos["Diabetic"] == "2":
        riesgo += 1


    if datos["fumador"] == "1":
        riesgo += 1

    if datos["DiffWalking"] == "Yes":
        riesgo += 1


    if datos["PhysicalActivity"] == "No":
        riesgo += 1


    if datos["GenHealth"] == "Fair" or datos["GenHealth"] == "Poor":
        riesgo += 1

    if datos["sex"] == "male":
        riesgo += 1

    if riesgo >= 7:
        return "Riesgo ALTO de enfermedad cardíaca"
    elif riesgo >= 4:
        return "Riesgo MODERADO de enfermedad cardíaca"
    else:
        return "Riesgo BAJO de enfermedad cardíaca"

def evaluar_riesgo_ictus(datos):
    riesgo = 0
    edad = int(datos["age"])
    sexo = datos["sexo"].lower()
    bmi = calcular_bmi(float(datos["mass"]), float(datos["height"]))
    glucosa = datos["glucose"]
    hipertension = int(datos["hipertension"])
    enfermedad_corazon = int(datos["heart_disease"])
    fumador = datos["fumador"]


    if 45 <= edad < 55:
        riesgo += 1
    elif edad >= 65:
        riesgo += 2


    if hipertension == 1:
        riesgo += 2

    if enfermedad_corazon == 1:
        riesgo += 2


    if glucosa == "1":
        riesgo += 1
    elif glucosa == "2":
        riesgo += 2


    if 25 <= bmi < 30:
        riesgo += 1
    elif bmi >= 30:
        riesgo += 2


    if fumador == "1":
        riesgo += 1
    elif fumador == "2":
        riesgo += 2

    if sexo == "female":
        riesgo += 1


    if riesgo >= 7:
        return "Riesgo ALTO de ictus"
    elif riesgo >= 4:
        return "Riesgo MODERADO de ictus"
    else:
        return "Riesgo BAJO de ictus"


