# Identificación de patrones de trayectoria académica mediante aprendizaje automático no supervisado

## Descripción del proyecto

Proyecto de investigación desarrollado para identificar patrones de trayectoria académica mediante técnicas de aprendizaje automático no supervisado, utilizando registros académicos históricos de estudiantes de un Instituto de Educación Superior Pedagógico Público de Tarma.

Se evalúan cuatro algoritmos de agrupamiento: K-Means, K-Medoids, clustering jerárquico aglomerativo y HDBSCAN. Los resultados se comparan mediante el coeficiente de silueta, el índice Davies-Bouldin y criterios de selección multicriterio.

## Entorno de ejecución

El entorno experimental de referencia utiliza Windows 11 de 64 bits, Python 3.12.10 y las bibliotecas Python documentadas en `requirements.txt`.

Para instalar las dependencias en un entorno virtual de Windows, ejecutar desde la raíz del proyecto:

1. `py -3.12 -m venv venv`
2. `.\venv\Scripts\python.exe -m pip install -r requirements.txt`
3. `.\venv\Scripts\python.exe -m pip check`

Para iniciar Jupyter Notebook:

`.\venv\Scripts\python.exe -m jupyter notebook`

## Estructura del proyecto

- `notebooks/`: notebooks de extracción, preparación, agrupamiento y comparación.
- `data/raw/pdf_estudiantes/`: fichas académicas originales en PDF.
- `data/interim/`: registros extraídos y estructurados.
- `data/processed/`: trayectorias académicas y matriz estandarizada.
- `results/`: resultados de los algoritmos y evaluaciones.
- `tests/`: pruebas automatizadas.
- `requirements.txt`: dependencias del entorno experimental.

## Orden de ejecución

Los notebooks deben ejecutarse desde el directorio `notebooks/`, respetando las rutas relativas definidas en el código.

1. `01_extraccion_de_5_pdf.ipynb`: extracción preliminar de cinco fichas PDF. Su ejecución es opcional.
2. `02_extraccion_dataset_muestra_completa.ipynb`: extracción del conjunto completo de fichas académicas disponibles.
3. `03_construccion_trayectorias.ipynb`: construcción de características académicas por estudiante.
4. `04_analisis_exploratorio.ipynb`: análisis exploratorio y estandarización de las once características.
5. `05_kmeans.ipynb`: evaluación y aplicación de K-Means.
6. `06_kmedoids.ipynb`: evaluación y aplicación de K-Medoids.
7. `07_clustering_jerarquico.ipynb`: evaluación de agrupamiento jerárquico.
8. `08_hdbscan.ipynb`: evaluación y aplicación de HDBSCAN.
9. `09_comparacion_modelos.ipynb`: comparación de métricas, evaluación sobre la muestra común y puntuación multicriterio.

Los notebooks 05 a 08 utilizan la misma matriz estandarizada. El notebook 09 requiere que se hayan generado los resultados de los cuatro algoritmos.

## Archivos principales

**Entradas:**

- `data/raw/pdf_estudiantes/`: documentos académicos originales.
- `data/interim/registros_academicos_raw.csv`: registros extraídos.
- `data/processed/trayectorias_academicas.csv`: características académicas por estudiante.
- `data/processed/X_clustering.csv`: matriz estandarizada utilizada por los algoritmos.

**Salidas:**

- `results/kmeans/`: resultados de K-Means.
- `results/kmedoids/`: resultados de K-Medoids.
- `results/jerarquico/`: resultados del agrupamiento jerárquico.
- `results/hdbscan/`: resultados de HDBSCAN.
- `results/comparacion/`: métricas y comparación multicriterio.
- `results/reproducibilidad/`: evidencias y resultados de estabilidad experimental.

## Validación experimental

El conjunto experimental de referencia comprende 244 trayectorias académicas representadas mediante 11 características.

La configuración final documentada de HDBSCAN utiliza `min_cluster_size = 30`, `min_samples = 10`, distancia euclidiana y selección de clústeres EOM.

Como resultados de referencia, se obtuvieron dos clústeres de 35 y 166 estudiantes, 43 observaciones clasificadas como ruido, un coeficiente de silueta de 0.593727 y un índice Davies-Bouldin de 0.598189.

Estos valores corresponden al conjunto experimental utilizado y deben verificarse al reproducir el procedimiento.

## Pruebas y reproducibilidad

Para ejecutar la prueba automatizada del núcleo HDBSCAN desde la raíz del proyecto:

`.\venv\Scripts\python.exe -m unittest discover -s tests -v`

La prueba utiliza datos sintéticos y verifica la ejecución del notebook y la generación de resultados. No constituye una prueba completa de extracción desde los PDF institucionales.

Para ejecutar el experimento complementario de estabilidad:

`.\venv\Scripts\python.exe experimento_estabilidad_20_remuestreos.py`

Este experimento realiza 20 remuestreos y genera estadísticas agregadas de estabilidad en `results/reproducibilidad/`.

## Protección de datos

Los registros académicos originales y los archivos intermedios pueden contener información personal. Estos archivos deben mantenerse protegidos y fuera del repositorio público.

Los identificadores de estudiantes no se utilizan como características de entrada para el agrupamiento. No obstante, su exclusión del modelo no implica que todos los archivos del proyecto estén completamente anonimizados.

Antes de publicar notebooks o resultados, se deben revisar las salidas guardadas para evitar la exposición de identificadores o información académica individual.

## Alcances

La investigación tiene un carácter descriptivo y exploratorio. Los patrones identificados permiten analizar similitudes entre trayectorias académicas, pero no establecen relaciones causales ni predicen automáticamente el rendimiento futuro de los estudiantes.