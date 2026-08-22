# =============================================================
# PROYECTO INTEGRADOR M5 - MODELO DE RIESGO CREDITICIO
# =============================================================
# Modulo: model_monitoring.py
# Descripcion: Monitoreo del modelo en produccion. Compara los datos
#              historicos (referencia) contra los datos actuales para
#              detectar data drift, usando pruebas estadisticas. Genera
#              una tabla de metricas de drift por variable con alertas.
# Autor: Simon Bedoya
# Carrera: Data Science - Soy Henry
# =============================================================

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency

from ft_engineering import ingenieria_caracteristicas


# Umbrales de alerta para las metricas de drift
UMBRAL_PSI = 0.25          # PSI mayor a 0.25 indica drift significativo
UMBRAL_PVALOR = 0.05       # p-valor menor a 0.05 indica drift
UMBRAL_JS = 0.1            # Jensen-Shannon mayor a 0.1 indica drift


# =============================================================
# METRICAS DE DRIFT PARA VARIABLES NUMERICAS
# =============================================================
def calcular_ks(referencia, actual):
    """
    Prueba de Kolmogorov-Smirnov. Compara dos distribuciones numericas.
    Un p-valor bajo indica que las distribuciones son diferentes (drift).
    """
    estadistico, p_valor = ks_2samp(referencia.dropna(), actual.dropna())
    return estadistico, p_valor


def calcular_psi(referencia, actual, bins=10):
    """
    Population Stability Index. Mide cuanto cambio la distribucion de una
    variable entre dos periodos. Se divide el rango en bins y se compara
    la proporcion de datos en cada uno.
    """
    referencia = referencia.dropna()
    actual = actual.dropna()

    # Se definen los limites de los bins con base en la referencia
    limites = np.percentile(referencia, np.linspace(0, 100, bins + 1))
    limites[0] = -np.inf
    limites[-1] = np.inf

    # Proporcion de datos en cada bin
    prop_ref = np.histogram(referencia, bins=limites)[0] / len(referencia)
    prop_act = np.histogram(actual, bins=limites)[0] / len(actual)

    # Se evita division por cero con un valor minimo
    prop_ref = np.where(prop_ref == 0, 0.0001, prop_ref)
    prop_act = np.where(prop_act == 0, 0.0001, prop_act)

    # Formula del PSI
    psi = np.sum((prop_act - prop_ref) * np.log(prop_act / prop_ref))
    return psi


def calcular_jensen_shannon(referencia, actual, bins=10):
    """
    Divergencia de Jensen-Shannon. Mide la similitud entre dos
    distribuciones, de 0 (iguales) a 1 (totalmente distintas).
    """
    referencia = referencia.dropna()
    actual = actual.dropna()

    # Se usa un rango comun para ambas distribuciones
    rango_min = min(referencia.min(), actual.min())
    rango_max = max(referencia.max(), actual.max())
    limites = np.linspace(rango_min, rango_max, bins + 1)

    # Distribuciones de probabilidad normalizadas
    p = np.histogram(referencia, bins=limites)[0] / len(referencia)
    q = np.histogram(actual, bins=limites)[0] / len(actual)

    # Se evita el cero
    p = np.where(p == 0, 0.0001, p)
    q = np.where(q == 0, 0.0001, q)

    # Distribucion promedio
    m = (p + q) / 2

    # Divergencia KL de cada distribucion respecto a la promedio
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    # Jensen-Shannon es el promedio de ambas
    js = (kl_pm + kl_qm) / 2
    return js


# =============================================================
# METRICA DE DRIFT PARA VARIABLES CATEGORICAS
# =============================================================
def calcular_chi_cuadrado(referencia, actual):
    """
    Prueba Chi-cuadrado de independencia para variables categoricas.
    Compara las frecuencias de las categorias entre los dos periodos.
    Un p-valor bajo indica un cambio significativo en las proporciones.
    """
    # Se cuentan las frecuencias de cada categoria en ambos periodos
    frec_ref = referencia.value_counts()
    frec_act = actual.value_counts()

    # Se unen las categorias de ambos periodos en una tabla de contingencia
    categorias = sorted(set(frec_ref.index) | set(frec_act.index))
    tabla = pd.DataFrame({
        'referencia': [frec_ref.get(cat, 0) for cat in categorias],
        'actual': [frec_act.get(cat, 0) for cat in categorias]
    }, index=categorias)

    # Prueba chi-cuadrado sobre la tabla de contingencia
    chi2, p_valor, _, _ = chi2_contingency(tabla)
    return chi2, p_valor


# =============================================================
# DIVISION DE DATOS: HISTORICO VS ACTUAL
# =============================================================
def dividir_por_fecha(df, proporcion_historico=0.7):
    """
    Divide el dataset en dos periodos segun la fecha del prestamo:
    los mas antiguos como referencia (historico) y los mas recientes
    como actuales, simulando el escenario de monitoreo en produccion.
    """
    df = df.sort_values('fecha_prestamo')
    punto_corte = int(len(df) * proporcion_historico)

    historico = df.iloc[:punto_corte]
    actual = df.iloc[punto_corte:]

    print(f"Periodo historico: {len(historico)} registros "
          f"(hasta {historico['fecha_prestamo'].max().date()})")
    print(f"Periodo actual: {len(actual)} registros "
          f"(desde {actual['fecha_prestamo'].min().date()})")

    return historico, actual


# =============================================================
# FUNCION PRINCIPAL DE MONITOREO
# =============================================================
def monitorear_drift(proporcion_historico=0.7):
    """
    Ejecuta el monitoreo completo: divide los datos por fecha, calcula
    las metricas de drift para cada variable y genera una tabla con
    las alertas correspondientes.
    """
    # Se cargan los datos con la ingenieria de caracteristicas
    df = ingenieria_caracteristicas()

    # Se dividen en historico y actual
    historico, actual = dividir_por_fecha(df, proporcion_historico)

    # Variables a monitorear
    numericas = ['edad_cliente', 'salario_cliente', 'puntaje',
                 'puntaje_datacredito', 'capital_prestado', 'cuota_pactada',
                 'plazo_meses', 'saldo_total']
    categoricas = ['tipo_laboral', 'tipo_credito', 'tendencia_ingresos']

    resultados = []

    # Drift de variables numericas
    for col in numericas:
        ks_stat, ks_p = calcular_ks(historico[col], actual[col])
        psi = calcular_psi(historico[col], actual[col])
        js = calcular_jensen_shannon(historico[col], actual[col])

        # Criterio de alerta basado en la magnitud del cambio (estandar en
        # riesgo crediticio): el PSI es la metrica principal de decision.
        # Un PSI alto indica drift significativo; un PSI moderado se confirma
        # con el KS test. El KS por si solo no dispara alerta, ya que con
        # muestras grandes detecta diferencias minimas sin relevancia practica.
        if psi > UMBRAL_PSI:
            hay_drift = True
        elif psi > 0.1 and ks_p < UMBRAL_PVALOR:
            hay_drift = True
        else:
            hay_drift = False

        resultados.append({
            'Variable': col,
            'Tipo': 'Numerica',
            'KS_pvalor': round(ks_p, 4),
            'PSI': round(psi, 4),
            'Jensen_Shannon': round(js, 4),
            'Chi2_pvalor': None,
            'Alerta': 'SI' if hay_drift else 'NO'
        })

    # Drift de variables categoricas
    for col in categoricas:
        chi2, chi_p = calcular_chi_cuadrado(historico[col], actual[col])

        hay_drift = chi_p < UMBRAL_PVALOR

        resultados.append({
            'Variable': col,
            'Tipo': 'Categorica',
            'KS_pvalor': None,
            'PSI': None,
            'Jensen_Shannon': None,
            'Chi2_pvalor': round(chi_p, 4),
            'Alerta': 'SI' if hay_drift else 'NO'
        })

    tabla_drift = pd.DataFrame(resultados)
    return tabla_drift, historico, actual

# =============================================================
# ANALISIS TEMPORAL: DRIFT POR PERIODOS
# =============================================================
def drift_temporal(n_periodos=4):
    """
    Divide el periodo actual en varios subperiodos por fecha y calcula
    el PSI promedio de las variables numericas en cada uno, respecto al
    historico. Permite observar la evolucion del drift en el tiempo.
    """
    df = ingenieria_caracteristicas()
    df = df.sort_values('fecha_prestamo')

    # El primer 70% es el historico de referencia
    punto_corte = int(len(df) * 0.7)
    historico = df.iloc[:punto_corte]
    actual = df.iloc[punto_corte:]

    numericas = ['edad_cliente', 'salario_cliente', 'puntaje',
                 'puntaje_datacredito', 'capital_prestado', 'cuota_pactada',
                 'plazo_meses', 'saldo_total']

    # Se divide el periodo actual en n subperiodos iguales
    actual = actual.reset_index(drop=True)
    tam_periodo = len(actual) // n_periodos

    evolucion = []
    for i in range(n_periodos):
        inicio = i * tam_periodo
        fin = (i + 1) * tam_periodo if i < n_periodos - 1 else len(actual)
        subperiodo = actual.iloc[inicio:fin]

        # PSI promedio de todas las numericas en este subperiodo
        psi_promedio = np.mean([
            calcular_psi(historico[col], subperiodo[col]) for col in numericas
        ])

        fecha_media = subperiodo['fecha_prestamo'].median()
        evolucion.append({
            'Periodo': f'P{i + 1}',
            'Fecha_media': fecha_media.date(),
            'PSI_promedio': round(psi_promedio, 4)
        })

    return pd.DataFrame(evolucion)


if __name__ == "__main__":
    tabla, historico, actual = monitorear_drift()
    print("\n" + "=" * 70)
    print("REPORTE DE DATA DRIFT")
    print("=" * 70)
    print(tabla.to_string(index=False))

    # Resumen de alertas
    n_alertas = (tabla['Alerta'] == 'SI').sum()
    print(f"\nVariables con drift detectado: {n_alertas} de {len(tabla)}")
    if n_alertas > 0:
        print("Recomendacion: revisar las variables con alerta y considerar reentrenar el modelo.")
    else:
        print("El modelo se mantiene estable. No se requiere accion inmediata.")