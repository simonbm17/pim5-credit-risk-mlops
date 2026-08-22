# =============================================================
# PROYECTO INTEGRADOR M5 - MODELO DE RIESGO CREDITICIO
# =============================================================
# Modulo: cargar_datos.py
# Descripcion: Carga de la base de datos cruda de creditos.
#              Lee el archivo fuente y lo entrega como DataFrame
#              para las etapas posteriores del pipeline.
# Autor: Simon Bedoya
# Carrera: Data Science - Soy Henry
# =============================================================

import pandas as pd
from pathlib import Path


def cargar_datos(ruta="Base_de_datos.xlsx"):
    """
    Carga la base de datos de creditos desde un archivo Excel.

    En un entorno productivo, esta informacion provendria del
    Data Warehouse o Data Lake de la empresa. Para este ejercicio
    se utiliza un dataset de ejemplo en formato Excel.

    Parametros
    ----------
    ruta : str
        Ruta al archivo con los datos crudos.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con los datos cargados sin transformar.
    """
    # Se construye la ruta del archivo de forma segura
    archivo = Path(ruta)

    # Se valida que el archivo exista antes de intentar leerlo
    if not archivo.exists():
        raise FileNotFoundError(f"No se encontro el archivo en la ruta: {ruta}")

    # Se lee el archivo Excel y se carga en un DataFrame
    df = pd.read_excel(archivo)

    # Se informa el resultado de la carga
    print(f"Datos cargados correctamente: {df.shape[0]} filas y {df.shape[1]} columnas.")

    return df


# Bloque de ejecucion directa: solo corre si se ejecuta este archivo
# de forma independiente, no cuando se importa desde otro modulo.
if __name__ == "__main__":
    datos = cargar_datos()
    print(datos.head())