# 🔍 Clustering y Detección de Anomalías con Scikit-Learn

Este proyecto explora diferentes técnicas de **clustering no supervisado** utilizando `scikit-learn`. Incluye implementaciones, visualizaciones y evaluaciones de rendimiento para algoritmos como:

- K-Means
- Mini-Batch KMeans
- DBSCAN
- Gaussian Mixture Models (GMM)

## 📊 Algoritmos implementados

### ✅ K-Means
- Entrenamiento en datasets sintéticos (`make_blobs`)
- Visualización de fronteras de decisión
- Análisis con el coeficiente de silueta
- Proceso iterativo paso a paso

### 🧠 Mini-Batch KMeans
- Entrenamiento con el dataset MNIST
- Manejo de mini-batches personalizados
- Evaluación del modelo usando inercia y `silhouette_score`

### 🌙 DBSCAN
- Agrupamiento en datasets no lineales (`make_moons`)
- Comparación con distintos valores de `eps`
- Detección de anomalías
- Clasificación de nuevos puntos con `KNeighborsClassifier`

### 🔮 Gaussian Mixture Models (GMM)
- Clustering probabilístico
- Visualización de densidades
- Detección de anomalías basada en distribución gaussiana

## 📦 Requisitos

Instala los paquetes necesarios con:

```bash
pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
