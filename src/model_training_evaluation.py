# =============================================================
# PROYECTO INTEGRADOR M5 - MODELO DE RIESGO CREDITICIO
# =============================================================
# Modulo: model_training_evaluation.py
# Descripcion: Entrenamiento y evaluacion de modelos supervisados.
#              Construye el pipeline de preprocesamiento con
#              ColumnTransformer, entrena varios modelos, los evalua
#              con metricas enfocadas en la clase minoritaria (los
#              clientes que no pagan a tiempo), selecciona el de
#              mejor desempenio y guarda el modelo final para su
#              posterior despliegue en la API.
# Autor: Simon Bedoya
# Carrera: Data Science - Soy Henry
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ft_engineering import ingenieria_caracteristicas


# =============================================================
# DEFINICION DE COLUMNAS POR TIPO
# =============================================================
# Variables numericas (incluye derivadas y transformaciones log).
# Se excluyen las monetarias originales, reemplazadas por su version log.
COLUMNAS_NUMERICAS = [
    'plazo_meses', 'edad_cliente', 'total_otros_prestamos', 'puntaje',
    'puntaje_datacredito', 'cant_creditosvigentes', 'huella_consulta',
    'saldo_mora', 'saldo_mora_codeudor', 'creditos_sectorFinanciero',
    'creditos_sectorCooperativo', 'creditos_sectorReal',
    'promedio_ingresos_datacredito', 'razon_cuota_capital',
    'razon_endeudamiento', 'tiene_mora', 'antiguedad_prestamo_dias',
    'capital_prestado_log', 'cuota_pactada_log', 'salario_cliente_log',
    'saldo_total_log', 'saldo_principal_log'
]

# Variables categoricas nominales (sin orden)
COLUMNAS_NOMINALES = ['tipo_laboral', 'tipo_credito']

# Variable categorica ordinal (con orden logico)
COLUMNAS_ORDINALES = ['tendencia_ingresos']

# Orden de las categorias para la variable ordinal
ORDEN_TENDENCIA = [['Sin_dato', 'Decreciente', 'Estable', 'Creciente']]

# Ruta donde se guarda el modelo final entrenado
RUTA_MODELO = "modelo_riesgo_credito.joblib"


# =============================================================
# CONSTRUCCION DEL PREPROCESADOR (ColumnTransformer, 3 ramas)
# =============================================================
def construir_preprocesador():
    """
    Construye el ColumnTransformer con tres ramas de procesamiento:
    numerica, categorica nominal y categorica ordinal.
    """
    # Rama numerica: imputa con la mediana y escala
    rama_numerica = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Rama categorica nominal: imputa con la mas frecuente y aplica OneHot
    rama_nominal = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Rama categorica ordinal: imputa con 'Sin_dato' y aplica orden
    rama_ordinal = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Sin_dato')),
        ('ordinal', OrdinalEncoder(categories=ORDEN_TENDENCIA,
                                   handle_unknown='use_encoded_value',
                                   unknown_value=-1))
    ])

    # Se combinan las tres ramas en un ColumnTransformer
    preprocesador = ColumnTransformer(transformers=[
        ('numeric', rama_numerica, COLUMNAS_NUMERICAS),
        ('nominal', rama_nominal, COLUMNAS_NOMINALES),
        ('ordinal', rama_ordinal, COLUMNAS_ORDINALES)
    ])

    return preprocesador


# =============================================================
# FUNCION PARA CONSTRUIR UN MODELO (pipeline completo)
# =============================================================
def build_model(estimador):
    """
    Construye un pipeline completo que une el preprocesamiento con
    un modelo estimador dado. Permite entrenar distintos modelos de
    forma consistente y reproducible.
    """
    modelo = Pipeline(steps=[
        ('preprocesador', construir_preprocesador()),
        ('clasificador', estimador)
    ])
    return modelo


# =============================================================
# FUNCION PARA RESUMIR EL DESEMPENIO DE UN MODELO
# =============================================================
def summarize_classification(y_true, y_pred, y_proba):
    """
    Calcula las metricas de evaluacion enfocadas en la clase minoritaria
    (clase 0 = no pago a tiempo), que es la de interes para el negocio.
    Con pos_label=0, el recall mide cuantos malos pagadores se detectan.
    """
    metricas = {
        'Exactitud': accuracy_score(y_true, y_pred),
        'Precision_clase0': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        'Recall_clase0': recall_score(y_true, y_pred, pos_label=0),
        'F1_clase0': f1_score(y_true, y_pred, pos_label=0),
        'ROC_AUC': roc_auc_score(y_true, y_proba)
    }
    return metricas


# =============================================================
# FUNCION PARA MOSTRAR LA MATRIZ DE CONFUSION
# =============================================================
def mostrar_matriz_confusion(y_true, y_pred, nombre):
    """Muestra la matriz de confusion de un modelo de forma legible."""
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nMatriz de confusion - {nombre}:")
    print(f"                 Pred: No pago   Pred: Pago")
    print(f"Real: No pago         {cm[0][0]:5d}        {cm[0][1]:5d}")
    print(f"Real: Pago            {cm[1][0]:5d}        {cm[1][1]:5d}")


# =============================================================
# FLUJO PRINCIPAL DE ENTRENAMIENTO Y EVALUACION
# =============================================================
def entrenar_y_evaluar():
    """
    Ejecuta el flujo completo: carga de datos, division train/test,
    entrenamiento de varios modelos, evaluacion, seleccion del mejor
    y guardado del modelo final para su despliegue.
    """
    # Se cargan los datos con la ingenieria de caracteristicas aplicada
    df = ingenieria_caracteristicas()

    # Se define la variable objetivo y las predictoras
    columnas_modelo = COLUMNAS_NUMERICAS + COLUMNAS_NOMINALES + COLUMNAS_ORDINALES
    X = df[columnas_modelo]
    y = df['Pago_atiempo']

    # Division estratificada, para conservar la proporcion de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    print(f"Entrenamiento: {X_train.shape[0]} registros | Prueba: {X_test.shape[0]} registros")
    print(f"Proporcion de la clase minoritaria en entrenamiento: "
          f"{(y_train == 0).mean():.3f}\n")

    # Se definen los modelos a comparar, todos con manejo del desbalance
    modelos = {
        'Regresion Logistica': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced', n_estimators=200, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(
            scale_pos_weight=20, random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(
            class_weight='balanced', random_state=42, verbose=-1)
    }

    # Se entrena y evalua cada modelo, guardando los resultados
    resultados = {}
    predicciones_proba = {}

    for nombre, estimador in modelos.items():
        print(f"Entrenando: {nombre}...")
        modelo = build_model(estimador)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]

        resultados[nombre] = summarize_classification(y_test, y_pred, y_proba)
        predicciones_proba[nombre] = y_proba
        mostrar_matriz_confusion(y_test, y_pred, nombre)

    # Se arma la tabla resumen de resultados
    tabla_resultados = pd.DataFrame(resultados).T
    tabla_resultados = tabla_resultados.round(4)
    print("\n" + "=" * 60)
    print("TABLA RESUMEN DE EVALUACION")
    print("=" * 60)
    print(tabla_resultados)

    # Se selecciona el mejor modelo segun el F1 de la clase minoritaria,
    # metrica que equilibra la deteccion de malos pagadores y las falsas alarmas
    mejor_modelo = tabla_resultados['F1_clase0'].idxmax()
    print(f"\nMejor modelo segun F1 de la clase minoritaria: {mejor_modelo}")

    # Se reentrena el mejor modelo con todo el conjunto de entrenamiento
    # y se guarda en disco para su posterior despliegue en la API
    modelo_final = build_model(modelos[mejor_modelo])
    modelo_final.fit(X_train, y_train)
    joblib.dump(modelo_final, RUTA_MODELO)
    print(f"Modelo final ({mejor_modelo}) guardado en: {RUTA_MODELO}")

    # Se generan los graficos comparativos
    graficar_comparacion(tabla_resultados)
    graficar_curvas_roc(y_test, predicciones_proba)

    return tabla_resultados


# =============================================================
# GRAFICOS COMPARATIVOS
# =============================================================
def graficar_comparacion(tabla):
    """Grafico de barras comparando las metricas de los modelos."""
    tabla[['Recall_clase0', 'F1_clase0', 'ROC_AUC', 'Precision_clase0']].plot(
        kind='bar', figsize=(12, 6))
    plt.title('Comparacion de metricas por modelo (clase minoritaria)')
    plt.ylabel('Valor')
    plt.xticks(rotation=15)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('comparacion_modelos.png', dpi=100)
    plt.show()


def graficar_curvas_roc(y_test, predicciones_proba):
    """Curvas ROC de todos los modelos en un mismo grafico."""
    plt.figure(figsize=(9, 7))
    for nombre, y_proba in predicciones_proba.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{nombre} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Azar')
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos (Recall)')
    plt.title('Curvas ROC comparativas')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('curvas_roc.png', dpi=100)
    plt.show()


if __name__ == "__main__":
    resultados = entrenar_y_evaluar()