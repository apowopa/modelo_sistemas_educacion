# 🎓 Sistema de Recomendación de Materiales Educativos Personalizados

Proyecto de **Modelado de Sistemas** que segmenta estudiantes según su perfil académico y recomienda materiales educativos personalizados utilizando técnicas de Machine Learning.

## Autores

- **[Apolonio Cuevas Manuel]**
- **[Sansores Arjona Alejandro]**
- **[Cauich Cauich Manuel]**

## 📌 Descripción

El sistema analiza datos de rendimiento estudiantil para:

1. **Explorar y limpiar** el dataset, identificando variables clave como calificaciones, tiempo de estudio, asistencia y actividades.
2. **Segmentar estudiantes** en grupos homogéneos mediante K-Means, determinando el número óptimo de clusters con el método del codo.
3. **Visualizar perfiles** de cada segmento con gráficos de radar que resumen fortalezas y debilidades.
4. **Generar recomendaciones por grupo**, asignando materiales educativos según las características de cada cluster.
5. **Recomendar individualmente** materiales a cada estudiante usando un modelo KNN que identifica vecinos exitosos y sugiere qué hacer diferente.

Todo se presenta en una aplicación interactiva construida con **Streamlit**.

## 📊 Base de Datos

- **Fuente:** [Students Performance Dataset – Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset)
- **Registros:** 2,392 estudiantes
- **Variables clave:**

| Variable | Descripción |
|---|---|
| `GPA` | Promedio de calificaciones |
| `StudyTimeWeekly` | Horas de estudio por semana |
| `Absences` | Número de ausencias |
| `Tutoring` | Si recibe tutoría (0/1) |
| `Extracurricular` | Participación extracurricular (0/1) |
| `Sports` | Participación deportiva (0/1) |
| `ParentalSupport` | Nivel de apoyo parental (0–4) |
| `GradeClass` | Clasificación de grado (A–F) |

## 🏗️ Estructura del Proyecto

```
├── app.py                  # Aplicación Streamlit (interfaz principal)
├── pyproject.toml           # Dependencias del proyecto
├── README.md
├── data/
│   ├── raw/                 # Dataset original (CSV)
│   └── clean/               # Dataset procesado
├── notebooks/
│   └── Avance_Proyecto_Modelado_de_sistemas.ipynb  # Exploración inicial
└── src/
    ├── __init__.py
    ├── limpieza.py          # Carga, exploración y limpieza de datos
    └── analisis.py          # Segmentación, radar charts, recomendaciones y modelo KNN
```

## 🧩 Módulos

### `src/limpieza.py`

- Descarga y extracción del dataset desde Kaggle.
- Carga del CSV crudo (`load_raw_data`).
- Exploración automática: dimensiones, tipos, nulos, estadísticas descriptivas, distribuciones categóricas y correlaciones (`explore_data`).
- Codificación one-hot y escalado estándar para el pipeline de limpieza.
- Mapas de valores categóricos a etiquetas legibles (`CATEGORY_MAPS`).

### `src/analisis.py`

- **Segmentación:** preparación de perfiles, método del codo, clustering K-Means (`segment_students`).
- **Perfiles de clusters:** resumen estadístico y etiquetas descriptivas automáticas (`label_clusters`).
- **Gráficos de radar:** normalización Min-Max con inversión de ausencias a "Asistencia", visualización individual y comparativa (`plot_radar_chart`, `plot_all_radars`).
- **Recomendaciones por grupo:** catálogo de 10 materiales educativos, asignación basada en reglas por las características del cluster (`get_all_cluster_recommendations`).
- **Modelo KNN:** encuentra los vecinos más similares con mejor GPA y sugiere mejoras específicas con justificación (`recommend_for_student`).

### `app.py`

Aplicación Streamlit con 4 pestañas:

| Pestaña | Contenido |
|---|---|
| 📊 Exploración de Datos | Vista previa, estadísticas, histogramas, correlación, variables categóricas |
| 🎯 Segmentación | Método del codo, boxplots por cluster, gráficos radar |
| 📋 Recomendaciones por Grupo | Perfil y materiales sugeridos para cada segmento |
| 🤖 Recomendación Individual | Selección de estudiante, perfil detallado y recomendaciones ML personalizadas |

## ⚙️ Instalación y Ejecución

### Requisitos

- Python ≥ 3.12

### Instalación

```bash
# Clonar el repositorio
git clone <repo-url>
cd modelo_sistemas_educacion

# Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Ejecución

```bash
# Lanzar la aplicación
streamlit run app.py
```

La app se abrirá en `http://localhost:8501`.

## 📦 Dependencias

| Paquete | Uso |
|---|---|
| `pandas` | Manipulación de datos |
| `scikit-learn` | K-Means, KNN, escalado |
| `streamlit` | Interfaz web interactiva |
| `seaborn` | Visualizaciones estadísticas |
| `matplotlib` | Gráficos de radar y plots |
| `requests` | Descarga del dataset |
