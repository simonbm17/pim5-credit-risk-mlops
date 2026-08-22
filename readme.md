# Modelo de Riesgo Crediticio con Enfoque MLOps

Sistema de predicción del comportamiento de pago de clientes de crédito, desarrollado con un enfoque completo de MLOps que abarca el análisis exploratorio de datos, la ingeniería de características, el modelamiento supervisado, el monitoreo del modelo en producción y su despliegue mediante una API.

## Caso de negocio

Las entidades financieras enfrentan el reto de estimar el riesgo asociado a cada solicitud de crédito antes de aprobarla. Un error en esta estimación tiene consecuencias directas sobre el negocio: aprobar un crédito a un cliente que no pagará genera pérdidas, mientras que rechazar a un buen cliente representa una oportunidad desperdiciada.

Este proyecto, desarrollado desde el rol de Científico de Datos Junior Advanced en una entidad financiera, aborda ese reto mediante un modelo de aprendizaje automático que predice si un nuevo cliente pagará su crédito a tiempo. El objetivo es apoyar la decisión de otorgamiento priorizando la detección de los clientes con mayor probabilidad de incumplimiento, que son los que representan el riesgo real para la cartera.

## Hallazgos clave

Tres descubrimientos del análisis marcaron el rumbo del proyecto y merecen destacarse.

La variable puntaje parecía inútil en un primer vistazo: casi todos los clientes tenían el mismo valor, cercano a 95, lo que la hacía ver como una constante sin capacidad de discriminar. El análisis bivariable reveló lo contrario. Los clientes que no pagan tienen un puntaje promedio de 23, frente a 94,5 de los que sí pagan, una diferencia superior al 300%. La aparente constancia escondía que justamente los pocos casos con puntaje bajo eran los malos pagadores. Una variable descartable en el análisis superficial resultó ser el predictor más poderoso del modelo.

Varias variables contenían valores centinela que, de no detectarse, habrían envenenado el modelo. La variable puntaje_datacredito presentaba ceros que no eran puntajes reales bajos, sino un código para clientes sin consulta de datos. De forma similar, puntaje tenía valores negativos que representaban registros sin evaluación. Tratados como números reales, estos valores habrían distorsionado el aprendizaje del modelo. Su identificación permitió convertirlos a nulos y manejarlos correctamente en la imputación.

La primera evaluación de los modelos arrojó un recall del 100%, un resultado sospechosamente perfecto. La investigación reveló que la métrica estaba midiendo la clase mayoritaria (los clientes que pagan), que es fácil de predecir por representar el 95% de los datos. Al corregir la evaluación para medir la clase minoritaria (los que no pagan, que son el verdadero interés del negocio), las métricas reflejaron el desempeño real del modelo. Este ajuste transformó un resultado engañoso en una evaluación honesta y confiable.

## Estructura del repositorio

    pim5-credit-risk-mlops/
    ├── src/
    │   ├── cargar_datos.py                 # Carga de la base de datos cruda
    │   ├── comprension_eda.ipynb           # Análisis exploratorio de datos
    │   ├── ft_engineering.py               # Ingeniería de características
    │   ├── model_training_evaluation.py    # Entrenamiento y evaluación de modelos
    │   ├── model_monitoring.py             # Monitoreo y detección de data drift
    │   ├── app_monitoring.py               # Aplicación Streamlit de monitoreo
    │   └── model_deploy.py                 # Despliegue del modelo (API)
    ├── Base_de_datos.xlsx                  # Dataset del proyecto
    ├── requirements.txt                    # Dependencias del proyecto
    ├── comparacion_modelos.png             # Gráfico comparativo de modelos
    ├── curvas_roc.png                      # Curvas ROC de los modelos
    └── readme.md                           # Documentación del proyecto

## El dataset

El conjunto de datos contiene 10.763 registros y 23 variables que describen a los clientes y sus créditos. La variable objetivo es Pago_atiempo, que indica si el cliente pagó a tiempo (1) o no (0).

Una característica central del problema es el fuerte desbalance de clases: el 95% de los registros corresponde a clientes que pagaron y solo el 5% a clientes que no pagaron. Esta condición determina las decisiones metodológicas de todo el proyecto, ya que la clase de interés es la minoritaria.

## Proceso y metodología

### 1. Carga de datos

El módulo cargar_datos.py se encarga de leer la base de datos desde el archivo fuente y entregarla como un DataFrame para las etapas posteriores. Incluye la validación de la existencia del archivo y reporta el número de registros y variables cargadas. En un entorno productivo real, esta información provendría del Data Warehouse de la empresa.

### 2. Análisis exploratorio de datos

El análisis exploratorio abarcó los niveles univariable, bivariable y multivariable, con interpretaciones en cada paso.

En el nivel univariable se caracterizaron los tipos de variables y se evaluó la calidad de los datos. Se detectaron valores imposibles, como una edad de 121 años, y valores centinela que representan datos faltantes. El análisis de asimetría y curtosis, junto con los histogramas y diagramas de caja, mostró que las variables monetarias presentan sesgo positivo extremo con valores atípicos severos. La variable salario_cliente resultó la más problemática, con valores que alcanzaban cifras irreales.

En el nivel bivariable se estudió la relación de cada variable con el objetivo. El puntaje se confirmó como el predictor más fuerte. La tendencia de ingresos mostró un patrón coherente: los clientes con ingresos decrecientes presentan la mayor tasa de no pago (6,3%) frente a los de ingresos crecientes (3,9%). Se aplicó criterio para no sobreinterpretar categorías con muy pocos registros, como algunos tipos de crédito con uno o dos casos.

En el nivel multivariable, la matriz de correlación reveló una única relación fuerte: capital_prestado y cuota_pactada, con una correlación de 0,76, coherente con su dependencia directa. El resto de las variables mostró independencia entre sí, lo que descarta multicolinealidad generalizada.

### 3. Ingeniería de características

A partir de los hallazgos del análisis se aplicaron reglas de validación que convierten los valores imposibles y los centinelas a nulos, conservando el resto de la información de cada registro. Se corrigieron los tipos de datos según la naturaleza de cada variable.

Se crearon cuatro variables derivadas: la razón cuota sobre capital, la razón de endeudamiento, un indicador de mora y la antigüedad del préstamo. A las variables monetarias con sesgo extremo se les aplicó una transformación logarítmica mediante log1p, que maneja correctamente los valores en cero.

La imputación y la codificación se trasladaron al pipeline de preprocesamiento. Esta decisión evita procesar los datos dos veces y garantiza que las mismas transformaciones se apliquen de forma idéntica durante el entrenamiento y durante la predicción en producción.

### 4. Modelamiento supervisado

El preprocesamiento se estructuró en un ColumnTransformer con tres ramas independientes: una numérica, con imputación por mediana y escalado; una categórica nominal, con imputación por moda y codificación One-Hot; y una categórica ordinal, con imputación por constante y codificación ordinal para tendencia_ingresos, respetando su orden lógico.

La imputación por mediana se eligió sobre la media por su robustez ante los valores atípicos detectados en el análisis. El tratamiento diferenciado de las variables categóricas nominales y ordinales respeta la naturaleza de cada una.

Se entrenaron y compararon cuatro modelos, todos con manejo del desbalance de clases: Regresión Logística, Random Forest, XGBoost y LightGBM. La evaluación se enfocó en la clase minoritaria, utilizando precisión, recall, F1 y ROC-AUC.

| Modelo              | Precisión (clase 0) | Recall (clase 0) | F1 (clase 0) | ROC-AUC |
| ------------------- | ------------------- | ---------------- | ------------ | ------- |
| Regresión Logística | 0.48                | 0.77             | 0.59         | 0.94    |
| Random Forest       | 1.00                | 0.73             | 0.84         | 0.96    |
| XGBoost             | 1.00                | 0.71             | 0.83         | 0.94    |
| LightGBM            | 0.94                | 0.74             | 0.82         | 0.95    |

El modelo seleccionado fue Random Forest, por presentar el mejor F1 de la clase de interés y el mejor ROC-AUC. La selección se basó en el F1 y no solo en el recall, porque el F1 equilibra la detección de malos pagadores con la ausencia de falsas alarmas. La Regresión Logística lograba un recall algo mayor, pero a costa de clasificar erróneamente a 86 buenos clientes como riesgosos, un comportamiento poco deseable para el negocio. El Random Forest detecta cerca del 73% de los clientes que no pagan sin generar una sola falsa alarma.

Se reconoce con transparencia que un recall del 73% implica que una parte de los malos pagadores no se detecta. Esto es esperable dada la dificultad de identificar a una clase minoritaria del 5%, y constituye una línea de mejora futura mediante técnicas como el ajuste del umbral de decisión o el sobremuestreo.

### 5. Monitoreo y detección de data drift

Para asegurar que el modelo mantenga su desempeño en el tiempo, se implementó un sistema de monitoreo que detecta cambios en la distribución de los datos. El monitoreo compara un periodo histórico de referencia contra un periodo actual, dividiendo los datos por la fecha del préstamo para simular el escenario real de producción, donde el drift ocurre con el paso del tiempo.

Se calcularon cuatro métricas de drift: Kolmogorov-Smirnov y divergencia de Jensen-Shannon para comparar distribuciones numéricas, Population Stability Index para medir la magnitud del cambio con los umbrales estándar de la industria financiera, y Chi-cuadrado para evaluar cambios en las proporciones de las variables categóricas.

Un aspecto metodológico importante define el criterio de alerta. La prueba de Kolmogorov-Smirnov, con muestras grandes, detecta como significativas diferencias mínimas sin relevancia práctica. Por esta razón, el criterio de alerta se basa en la magnitud del cambio medida por el PSI, y no únicamente en la significancia estadística. Bajo este criterio robusto, se detectó drift en tres variables: plazo_meses, tipo_credito y tendencia_ingresos.

El sistema incluye una aplicación en Streamlit que presenta un semáforo de alertas según el nivel de riesgo, la tabla de métricas por variable, la comparación visual entre distribuciones históricas y actuales, el análisis de la evolución del drift a lo largo de subperiodos y recomendaciones automáticas que sugieren revisar variables o reentrenar el modelo según la gravedad detectada.

## Cómo ejecutar el proyecto

1. Clonar el repositorio:

   git clone https://github.com/simonbm17/pim5-credit-risk-mlops.git

   cd pim5-credit-risk-mlops

2. Crear y activar un entorno virtual:

   python -m venv venv
   venv\Scripts\activate

3. Instalar las dependencias:

   pip install -r requirements.txt

4. Ejecutar el entrenamiento y la evaluación de los modelos. Este paso genera el modelo entrenado y los gráficos comparativos:

   python src\model_training_evaluation.py

5. Ejecutar el monitoreo de data drift, que genera el reporte de métricas por variable:

   python src\model_monitoring.py

6. Levantar la aplicación de monitoreo en Streamlit, que abre el panel visual en el navegador:

   streamlit run src\app_monitoring.py

## Versionamiento

El proyecto sigue un flujo de trabajo con tres ramas: developer para el desarrollo activo, certification como entorno de validación y main para las versiones estables. El versionamiento semántico registra la evolución del proyecto:

- v1.0.0: estructura inicial del proyecto.
- v1.0.1: carga de datos y análisis exploratorio.
- v1.1.0: ingeniería de características y modelamiento, integrada a través de la rama de certificación.
- v1.2.0: monitoreo de data drift y aplicación de visualización.

## Autor

Simón Bedoya. Data Science - Soy Henry
