# =============================================================
# PROYECTO INTEGRADOR M5 - MODELO DE RIESGO CREDITICIO
# =============================================================
# Modulo: app_monitoring.py
# Descripcion: Aplicacion en Streamlit para el monitoreo de data drift.
#              Presenta de forma visual las metricas de drift, la
#              comparacion entre distribuciones, el analisis temporal
#              y las recomendaciones automaticas.
# Autor: Simon Bedoya
# Carrera: Data Science - Soy Henry
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model_monitoring import (monitorear_drift, drift_temporal,
                              UMBRAL_PSI)

# Configuracion de la pagina
st.set_page_config(page_title="Monitoreo de Data Drift",
                   page_icon="grafico", layout="wide")

# Titulo principal
st.title("Monitoreo de Data Drift - Modelo de Riesgo Crediticio")
st.markdown("Panel de control para la deteccion de cambios en la poblacion "
            "que puedan afectar el desempenio del modelo en produccion.")

# Se calcula el drift (se cachea para no recalcular en cada interaccion)
@st.cache_data
def cargar_datos_monitoreo():
    tabla, historico, actual = monitorear_drift()
    evolucion = drift_temporal()
    return tabla, historico, actual, evolucion

with st.spinner("Calculando metricas de drift..."):
    tabla, historico, actual, evolucion = cargar_datos_monitoreo()

# =============================================================
# SECCION 1: SEMAFORO DE ALERTAS
# =============================================================
st.header("Estado general del monitoreo")

n_alertas = (tabla['Alerta'] == 'SI').sum()
n_total = len(tabla)

col1, col2, col3 = st.columns(3)
col1.metric("Variables monitoreadas", n_total)
col2.metric("Variables con drift", n_alertas)
col3.metric("Variables estables", n_total - n_alertas)

# Semaforo segun la cantidad de alertas, con colores explicitos
if n_alertas == 0:
    color_fondo = "#2e7d32"   # verde
    mensaje = "SEMAFORO VERDE: No se detecta drift significativo. El modelo se mantiene estable."
elif n_alertas <= 3:
    color_fondo = "#f9a825"   # amarillo
    mensaje = "SEMAFORO AMARILLO: Drift moderado en algunas variables. Se recomienda revisar."
else:
    color_fondo = "#c62828"   # rojo
    mensaje = "SEMAFORO ROJO: Drift significativo en varias variables. Se recomienda reentrenar el modelo."

st.markdown(
    f"""
    <div style="background-color: {color_fondo}; padding: 15px;
    border-radius: 8px; color: white; font-weight: bold; font-size: 16px;">
    {mensaje}
    </div>
    """,
    unsafe_allow_html=True
)

# =============================================================
# SECCION 2: TABLA DE METRICAS DE DRIFT
# =============================================================
st.header("Metricas de drift por variable")

# Se reemplazan los valores vacios (NaN) por un guion para mayor claridad
tabla_visual = tabla.fillna('—')

# Se resaltan las filas con alerta
def resaltar_alertas(fila):
    if fila['Alerta'] == 'SI':
        estilo = 'background-color: #ffcccc; color: #000000; font-weight: bold'
    else:
        estilo = 'background-color: #e8f5e9; color: #000000'
    return [estilo] * len(fila)

st.dataframe(tabla_visual.style.apply(resaltar_alertas, axis=1), use_container_width=True)

st.caption("KS_pvalor y Chi2_pvalor: valores menores a 0.05 indican diferencia. "
           "PSI: mayor a 0.25 indica drift significativo. "
           "Jensen-Shannon: mayor a 0.1 indica drift.")

# =============================================================
# SECCION 3: COMPARACION DE DISTRIBUCIONES
# =============================================================
st.header("Comparacion de distribuciones: historico vs actual")

numericas = ['edad_cliente', 'salario_cliente', 'puntaje',
             'puntaje_datacredito', 'capital_prestado', 'cuota_pactada',
             'plazo_meses', 'saldo_total']

variable_sel = st.selectbox("Selecciona una variable numerica:", numericas)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(historico[variable_sel].dropna(), bins=40, alpha=0.6,
        label='Historico', color='steelblue', density=True)
ax.hist(actual[variable_sel].dropna(), bins=40, alpha=0.6,
        label='Actual', color='orange', density=True)
ax.set_title(f'Distribucion de {variable_sel}')
ax.set_xlabel(variable_sel)
ax.set_ylabel('Densidad')
ax.legend()
st.pyplot(fig)

# Se muestra el PSI de la variable seleccionada
psi_var = tabla.loc[tabla['Variable'] == variable_sel, 'PSI'].values
if len(psi_var) > 0 and psi_var[0] is not None:
    st.info(f"PSI de {variable_sel}: {psi_var[0]} "
            f"({'Drift' if psi_var[0] > UMBRAL_PSI else 'Estable'})")

# =============================================================
# SECCION 4: ANALISIS TEMPORAL
# =============================================================
st.header("Evolucion del drift en el tiempo")
st.markdown("PSI promedio de las variables numericas a lo largo de los "
            "subperiodos recientes, respecto al periodo historico.")

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(evolucion['Periodo'], evolucion['PSI_promedio'],
         marker='o', color='darkred', linewidth=2)
ax2.axhline(y=0.1, color='orange', linestyle='--', label='Umbral moderado (0.1)')
ax2.axhline(y=0.25, color='red', linestyle='--', label='Umbral significativo (0.25)')
ax2.set_title('Evolucion del PSI promedio')
ax2.set_xlabel('Subperiodo')
ax2.set_ylabel('PSI promedio')
ax2.legend()
st.pyplot(fig2)

st.dataframe(evolucion, use_container_width=True)

# =============================================================
# SECCION 5: RECOMENDACIONES AUTOMATICAS
# =============================================================
st.header("Recomendaciones")

if n_alertas == 0:
    st.success("El modelo se mantiene estable. No se requiere accion inmediata. "
               "Se recomienda continuar con el monitoreo periodico.")
elif n_alertas <= 3:
    st.warning("Se detecta drift moderado. Recomendaciones:")
    st.markdown("""
    - Revisar en detalle las variables marcadas con alerta.
    - Analizar si los cambios responden a factores estacionales o estructurales.
    - Programar una evaluacion del desempenio del modelo con datos recientes.
    """)
    variables_alerta = tabla.loc[tabla['Alerta'] == 'SI', 'Variable'].tolist()
    st.markdown(f"**Variables a revisar:** {', '.join(variables_alerta)}")
else:
    st.error("Se detecta drift significativo. Recomendaciones:")
    st.markdown("""
    - Reentrenar el modelo con datos actualizados.
    - Revisar la ingenieria de caracteristicas de las variables afectadas.
    - Evaluar si el modelo actual sigue siendo valido para produccion.
    """)
    variables_alerta = tabla.loc[tabla['Alerta'] == 'SI', 'Variable'].tolist()
    st.markdown(f"**Variables con drift:** {', '.join(variables_alerta)}")

st.markdown("---")
st.caption("Proyecto Integrador M5 - Modelo de Riesgo Crediticio | Simon Bedoya")