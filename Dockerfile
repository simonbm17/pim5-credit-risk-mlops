# =============================================================
# PROYECTO INTEGRADOR M5 - MODELO DE RIESGO CREDITICIO
# Imagen Docker para el despliegue de la API de prediccion
# =============================================================

# Imagen base con Python 3.12
FROM python:3.12-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Se copian primero los requerimientos para aprovechar la cache de Docker
COPY requirements.txt .

# Se instalan las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Se copia el codigo fuente y el modelo entrenado
COPY src/ ./src/
COPY modelo_riesgo_credito.joblib .
COPY Base_de_datos.xlsx .

# Se expone el puerto donde correra la API
EXPOSE 8000

# Comando para iniciar la API con Uvicorn al arrancar el contenedor
CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]