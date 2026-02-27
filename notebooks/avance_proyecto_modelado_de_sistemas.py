import requests
import os
import pandas as pd
import 


DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/rabieelkharoua/students-performance-dataset"


df = pd.read_csv(file_path)

# Mostrar las primeras 5 filas del DataFrame
df.head()

"""## Exploración y limpieza inicial de datos

### Subtarea:
Examinar la estructura del dataset, identificar variables clave (puntuaciones, tiempo de estudio, asistencia, métodos de aprendizaje), comprobar valores faltantes y preparar los datos para el análisis y la aplicación del modelo.

"""

print(f"Dimensiones del DataFrame: {df.shape}\n")

print("Información del DataFrame:")
df.info()

print("\nEstadísticas descriptivas:")
print(df.describe())

print("\nValores faltantes por columna:")
print(df.isnull().sum())

categorical_cols = [
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
    "GradeClass",
]

print("\nValores únicos y conteos para columnas categóricas:")
for col in categorical_cols:
    if col in df.columns:
        print("\n--- {} ---".format(col))
        print(df[col].value_counts())
    else:
        print("La columna '{}' no se encontró en el DataFrame.".format(col))

"""## Ingeniería de características para perfiles de estudiantes

### Subtarea:
Crear o transformar características a partir de las variables clave para construir perfiles completos de estudiantes. Esto puede implicar codificar variables categóricas y normalizar datos numéricos.

"""

from sklearn.preprocessing import StandardScaler

# Aplicar codificación one-hot a LAS columnas categóricas
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Identificar las columnas numéricas.
numerical_cols = ["Age", "StudyTimeWeekly", "Absences", "GPA"]

# Ajustar y transformar las columnas numéricas
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# DataFrame final para perfiles de estudiantes
student_profiles_df = df_encoded.drop(columns=["StudentID"], errors="ignore")

print(
    "Dimensiones del DataFrame de perfiles de estudiantes:", student_profiles_df.shape
)

print("Primeras 5 filas del DataFrame de perfiles de estudiantes:")
student_profiles_df.head()

"""## Segmentar estudiantes según los perfiles

### Subtarea:
Aplicar algoritmos de clustering para agrupar estudiantes con características similares. Determinar el número óptimo de clusters mediante el método del codo (Elbow).

"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Método del codo para determinar k óptimo
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(student_profiles_df)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker="o")
plt.title("Método del codo para k óptimo")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inertia")
plt.xticks(k_range)
plt.grid(True)
plt.show()

print("Valores de inertia calculados para diferentes k.")

optimal_k = 4  # Elegimos 4 :D

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
student_clusters = kmeans_final.fit_predict(student_profiles_df)

# Asignar etiquetas de cluster al DataFrame original
df["Cluster"] = student_clusters

print(f"Asignadas {optimal_k} clusters al DataFrame.")

print("Primeras 5 filas del DataFrame con la nueva columna 'Cluster':")
df.head()

"""## Identificar áreas para recomendaciones personalizadas

### Subtarea:
Para cada segmento, identificar fortalezas y debilidades comunes y formular recomendaciones conceptuales de materiales educativos o áreas de enfoque.

"""

cluster_summary = df.groupby("Cluster").mean()
print("Media de características numéricas por cluster:")
print(cluster_summary[["Age", "StudyTimeWeekly", "Absences", "GPA"]])

# categorical_cols = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GradeClass']
print("\nModa de características categóricas por cluster:")
for col in categorical_cols:
    print("\n--- {} ---".format(col))
    print(
        df.groupby("Cluster")[col].agg(
            lambda x: x.mode()[0] if not x.mode().empty else "No mode"
        )
    )

"""### Análisis de segmentos y recomendaciones

Se identificaron cuatro segmentos de estudiantes con características distintas; a continuación se resumen perfiles y recomendaciones conceptuales para cada cluster.

**Cluster 0: Estudiantes mayores, buen rendimiento y ausencias moderadas**
- Perfil: Estudiantes de mayor edad (promedio ~17.5 años) con buen rendimiento académico (GPA ~2.7), tiempo de estudio moderado y ausencias moderadas (aprox. 7 por semestre).
- Fortalezas: Buen desempeño académico y hábitos de estudio estables.
- Debilidades: Ausencias moderadas que pueden indicar descompromiso ocasional.
- Recomendaciones: materiales de enriquecimiento, oportunidades de mentoría y seguimiento de asistencia.

**Cluster 1: Estudiantes jóvenes, bajo rendimiento y altas ausencias**
- Perfil: Estudiantes más jóvenes (promedio ~15.5 años) con bajo GPA (~1.18) y muchas ausencias (aprox. 21 por semestre).
- Fortalezas: Edad temprana, potencial para intervención.
- Debilidades: Bajo rendimiento y elevada absentismo.
- Recomendaciones: intervención personalizada, tutorías, acciones para mejorar la asistencia y mayor implicación familiar.

**Cluster 2: Estudiantes jóvenes, buen rendimiento y bajas ausencias**
- Perfil: Estudiantes jóvenes (promedio ~15.5 años) con buen GPA (~2.72) y bajas ausencias (aprox. 7).
- Fortalezas: Buen desempeño y hábito de estudio consistente.
- Debilidades: Pueden necesitar mayor desafío académico.
- Recomendaciones: programas de enriquecimiento, mentoría y actividades avanzadas.

**Cluster 3: Estudiantes mayores, bajo rendimiento y altas ausencias**
- Perfil: Estudiantes de mayor edad (promedio ~17.5 años) con bajo GPA (~1.16) y muchas ausencias (aprox. 21).
- Fortalezas: Posible receptividad a orientación vocacional.
- Debilidades: Riesgo de deserción por bajo rendimiento y absentismo.
- Recomendaciones: apoyo intensivo, opciones educativas flexibles y orientación profesional.

## Visualizar segmentos e insights

### Subtarea:
Crear visualizaciones que ilustren los diferentes segmentos, sus características clave y la justificación de las recomendaciones conceptuales. Incluir leyendas adecuadas.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Características numéricas a visualizar
features_to_plot = ["GPA", "Absences", "StudyTimeWeekly", "Age"]

# Crear una figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()  # Aplanar para iterar

for i, feature in enumerate(features_to_plot):
    sns.boxplot(
        x="Cluster", y=feature, data=df, ax=axes[i], palette="viridis", dodge=False
    )
    axes[i].set_title(f"{feature} por Cluster")
    axes[i].set_xlabel("Cluster")
    axes[i].set_ylabel(feature)
    axes[i].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
print("Se generaron los boxplots de las características numéricas por cluster.")

import matplotlib.pyplot as plt
import seaborn as sns

categorical_features_to_plot = [
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
    "GradeClass",
]

n_features = len(categorical_features_to_plot)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes = axes.flatten()

for i, feature in enumerate(categorical_features_to_plot):
    if i < len(axes):
        sns.countplot(x="Cluster", hue=feature, data=df, ax=axes[i], palette="tab10")
        axes[i].set_title(f"Distribución de {feature} por Cluster")
        axes[i].set_xlabel("Cluster")
        axes[i].set_ylabel("Conteo")
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].legend(title=feature)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
print("Se generaron los countplots de las características categóricas por cluster.")

"""## Resumen:

### Hallazgos clave del análisis de datos

- **Visión general**: El dataset contiene registros de estudiantes. Verifique el nombre y la ruta del CSV después de descomprimir.
- **Ingeniería de características**: Se aplicó one-hot encoding a variables categóricas y escalado a variables numéricas (Age, StudyTimeWeekly, Absences, GPA).
- **Segmentación**: Se utilizó el método del codo y K-Means para segmentar en 4 clusters.
"""
