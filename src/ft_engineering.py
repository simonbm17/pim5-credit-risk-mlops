# =============================================================
# PROYECTO INTEGRADOR M5 - MODELO DE RIESGO CREDITICIO
# =============================================================
# Modulo: ft_engineering.py
# Descripcion: Ingenieria de caracteristicas. Aplica las reglas de
#              validacion, la correccion de tipos, la creacion de
#              variables derivadas y las transformaciones definidas
#              en el EDA. La imputacion y codificacion se realizan
#              posteriormente en el pipeline de preprocesamiento.
# Autor: Simon Bedoya
# Carrera: Data Science - Soy Henry
# =============================================================

import pandas as pd
import numpy as np
from cargar_datos import cargar_datos

# Categorias validas para la variable tendencia_ingresos
CATEGORIAS_TENDENCIA = ['Creciente', 'Decreciente', 'Estable']


def validar_valores(df):
    """
    Aplica las reglas de validacion definidas en el EDA.
    Los valores imposibles o fuera de rango se convierten a nulos
    para su posterior imputacion en el pipeline, conservando la fila.
    """
    df = df.copy()

    # Edad: rango valido de 18 a 100 anios
    df.loc[(df['edad_cliente'] < 18) | (df['edad_cliente'] > 100), 'edad_cliente'] = np.nan

    # Puntaje: escala de 0 a 100, los negativos son codigos invalidos
    df.loc[(df['puntaje'] < 0) | (df['puntaje'] > 100), 'puntaje'] = np.nan

    # Puntaje datacredito: escala de 1 a 999, el 0 significa sin dato
    df.loc[(df['puntaje_datacredito'] <= 0) | (df['puntaje_datacredito'] > 999),
           'puntaje_datacredito'] = np.nan

    # Salario: debe ser mayor a 0 y menor a un tope razonable
    df.loc[(df['salario_cliente'] <= 0) | (df['salario_cliente'] > 1_000_000_000),
           'salario_cliente'] = np.nan

    # Tendencia de ingresos: solo se aceptan las tres categorias validas
    df['tendencia_ingresos'] = df['tendencia_ingresos'].where(
        df['tendencia_ingresos'].isin(CATEGORIAS_TENDENCIA), other=np.nan)

    return df


def corregir_tipos(df):
    """
    Convierte cada variable a su tipo de dato correcto segun su naturaleza.
    """
    df = df.copy()

    # tipo_credito representa codigos, no cantidades: se trata como categorica
    df['tipo_credito'] = df['tipo_credito'].astype('object')

    # Variable categorica de texto
    df['tipo_laboral'] = df['tipo_laboral'].astype('object')

    # La fecha se asegura como datetime
    df['fecha_prestamo'] = pd.to_datetime(df['fecha_prestamo'])

    return df


def crear_variables_derivadas(df):
    """
    Crea nuevas variables a partir de las existentes para aportar
    informacion util al modelo. Las operaciones toleran valores nulos,
    que seran imputados posteriormente en el pipeline.
    """
    df = df.copy()

    # Razon cuota sobre capital: mide el peso de la cuota frente al monto
    df['razon_cuota_capital'] = df['cuota_pactada'] / df['capital_prestado']

    # Razon de endeudamiento: carga de otros prestamos frente al salario
    # Se suma 1 para evitar division por cero cuando el salario es nulo o cero
    df['razon_endeudamiento'] = df['total_otros_prestamos'] / (df['salario_cliente'].fillna(0) + 1)

    # Indicador de mora: 1 si el cliente tiene algun saldo en mora
    df['tiene_mora'] = (df['saldo_mora'].fillna(0) > 0).astype(int)

    # Antiguedad del prestamo en dias desde la fecha mas reciente del dataset
    fecha_referencia = df['fecha_prestamo'].max()
    df['antiguedad_prestamo_dias'] = (fecha_referencia - df['fecha_prestamo']).dt.days

    return df


def transformar_variables(df):
    """
    Aplica transformacion logaritmica a las variables monetarias con
    sesgo extremo. Se usa log1p, que maneja los ceros. Los valores nulos
    permanecen nulos y se imputan despues en el pipeline.
    """
    df = df.copy()

    variables_log = ['capital_prestado', 'cuota_pactada', 'salario_cliente',
                    'saldo_total', 'saldo_principal']

    for col in variables_log:
        df[f'{col}_log'] = np.log1p(df[col])

    return df


def ingenieria_caracteristicas(ruta="Base_de_datos.xlsx"):
    """
    Funcion principal. Ejecuta el proceso de ingenieria de caracteristicas
    en orden y devuelve el DataFrame con las variables listas para el
    pipeline de preprocesamiento y modelado.
    """
    df = cargar_datos(ruta)

    df = validar_valores(df)
    df = corregir_tipos(df)
    df = crear_variables_derivadas(df)
    df = transformar_variables(df)

    print(f"Ingenieria de caracteristicas completada: {df.shape[0]} filas y {df.shape[1]} columnas.")

    return df

def procesar_dataframe(df):
    """
    Aplica la ingenieria de caracteristicas a un DataFrame ya cargado
    en memoria, sin leer el archivo Excel. Se usa en el despliegue,
    donde los datos llegan directamente en la peticion a la API.
    """
    df = validar_valores(df)
    df = corregir_tipos(df)
    df = crear_variables_derivadas(df)
    df = transformar_variables(df)
    return df
    
if __name__ == "__main__":
    datos = ingenieria_caracteristicas()
    print("\nPrimeras filas del dataset procesado:")
    print(datos.head())