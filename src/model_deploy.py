# =============================================================
# PROYECTO INTEGRADOR M5 - MODELO DE RIESGO CREDITICIO
# =============================================================
# Modulo: model_deploy.py
# Descripcion: Despliegue del modelo mediante una API REST construida
#              con FastAPI. Expone un endpoint /predict que recibe datos
#              crudos de uno o varios clientes, les aplica la ingenieria
#              de caracteristicas y retorna las predicciones de riesgo
#              crediticio. Soporta prediccion por lotes (batch).
# Autor: Simon Bedoya
# Carrera: Data Science - Soy Henry
# =============================================================

import os
import sys
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Se asegura que la carpeta actual (src) este en la ruta de busqueda de
# modulos, para que la importacion funcione tanto en local como en Docker.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ft_engineering import procesar_dataframe

# Ruta del modelo entrenado. Se construye de forma absoluta apuntando a
# la raiz del proyecto (un nivel arriba de src), para que la API encuentre
# el modelo sin importar desde que carpeta se ejecute.
RUTA_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(RUTA_BASE, "modelo_riesgo_credito.joblib")

# Se carga el modelo una sola vez al iniciar la aplicacion
modelo = joblib.load(RUTA_MODELO)

# Se crea la aplicacion FastAPI
app = FastAPI(
    title="API de Riesgo Crediticio",
    description="Servicio para predecir el comportamiento de pago de "
                "clientes de credito. Recibe datos crudos y retorna la "
                "prediccion de riesgo.",
    version="1.0.0"
)


# =============================================================
# ESQUEMA DE DATOS DE ENTRADA
# =============================================================
# Define la estructura de un cliente. Cada campo corresponde a una
# variable cruda del dataset original. Pydantic valida los tipos.
class Cliente(BaseModel):
    tipo_credito: int
    fecha_prestamo: str
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tendencia_ingresos: str


# Esquema para recibir un lote de clientes
class LoteClientes(BaseModel):
    clientes: List[Cliente]


# =============================================================
# ENDPOINT DE BIENVENIDA
# =============================================================
@app.get("/")
def inicio():
    """Endpoint raiz que confirma que la API esta activa."""
    return {"mensaje": "API de Riesgo Crediticio activa. "
                       "Use el endpoint /predict para obtener predicciones."}


# =============================================================
# ENDPOINT DE PREDICCION (soporta lotes)
# =============================================================
@app.post("/predict")
def predecir(lote: LoteClientes):
    """
    Recibe uno o varios clientes con sus datos crudos, aplica la
    ingenieria de caracteristicas y retorna las predicciones.
    Para cada cliente devuelve la prediccion (1 = paga a tiempo,
    0 = no paga) y la probabilidad de pago.
    """
    # Se convierten los datos recibidos en un DataFrame
    datos = pd.DataFrame([cliente.model_dump() for cliente in lote.clientes])

    # Se aplica la ingenieria de caracteristicas (mismos pasos del entrenamiento)
    datos_procesados = procesar_dataframe(datos)

    # Se generan las predicciones y las probabilidades
    predicciones = modelo.predict(datos_procesados)
    probabilidades = modelo.predict_proba(datos_procesados)[:, 1]

    # Se arma la respuesta para cada cliente
    resultados = []
    for i in range(len(predicciones)):
        resultados.append({
            "cliente": i + 1,
            "prediccion": int(predicciones[i]),
            "resultado": "Paga a tiempo" if predicciones[i] == 1 else "No paga a tiempo",
            "probabilidad_pago": round(float(probabilidades[i]), 4)
        })

    return {"predicciones": resultados}