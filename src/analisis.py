"""
Módulo de análisis: segmentación de estudiantes, gráficos radar
y modelo de recomendación de materiales educativos.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from src.limpieza import CATEGORICAL_COLS, NUMERICAL_COLS

# ─── Features para radar ────────────────────────────────────────────────────

RADAR_RAW_COLS = [
    "GPA",
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "Extracurricular",
    "Sports",
]

# ─── Catálogo de materiales educativos ───────────────────────────────────────

MATERIAL_CATALOG = {
    "study_guides": {
        "nombre": "📚 Guías de Estudio Estructuradas",
        "descripcion": "Planes de estudio semanales con técnicas de pomodoro y resúmenes.",
        "tipo": "Material de apoyo",
    },
    "tutoring": {
        "nombre": "👨‍🏫 Programa de Tutorías",
        "descripcion": "Sesiones de tutoría individual o grupal con seguimiento.",
        "tipo": "Intervención personalizada",
    },
    "attendance_tracker": {
        "nombre": "📅 Seguimiento de Asistencia",
        "descripcion": "App de recordatorios, alertas a padres, material de clase en línea.",
        "tipo": "Herramienta digital",
    },
    "advanced_materials": {
        "nombre": "🧠 Materiales Avanzados y Retos",
        "descripcion": "Proyectos de investigación, competencias académicas, olimpiadas.",
        "tipo": "Enriquecimiento",
    },
    "remedial_exercises": {
        "nombre": "📝 Ejercicios de Refuerzo",
        "descripcion": "Ejercicios adaptativos por nivel, práctica con retroalimentación.",
        "tipo": "Material de refuerzo",
    },
    "online_courses": {
        "nombre": "💻 Cursos en Línea Complementarios",
        "descripcion": "Videos, quizzes y foros en plataformas educativas.",
        "tipo": "Recurso digital",
    },
    "mentoring": {
        "nombre": "🤝 Programa de Mentoría",
        "descripcion": "Mentoría entre pares o con profesores para orientación académica.",
        "tipo": "Desarrollo personal",
    },
    "family_engagement": {
        "nombre": "👨‍👩‍👧 Involucramiento Familiar",
        "descripcion": "Talleres para padres, reportes de progreso, comunicación continua.",
        "tipo": "Apoyo social",
    },
    "extracurricular": {
        "nombre": "🎯 Actividades Extracurriculares",
        "descripcion": "Clubes, deportes, artes y voluntariado para desarrollo integral.",
        "tipo": "Desarrollo integral",
    },
    "vocational": {
        "nombre": "🎓 Orientación Vocacional",
        "descripcion": "Tests vocacionales, visitas a empresas, ferias de carreras.",
        "tipo": "Orientación",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  SEGMENTACIÓN
# ═══════════════════════════════════════════════════════════════════════════


def prepare_profiles(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Prepara perfiles codificados y escalados para clustering."""
    profiles = df.drop(columns=["StudentID"], errors="ignore").copy()
    profiles = pd.get_dummies(profiles, columns=CATEGORICAL_COLS, drop_first=False)
    scaler = StandardScaler()
    profiles[NUMERICAL_COLS] = scaler.fit_transform(profiles[NUMERICAL_COLS])
    return profiles, scaler


def find_optimal_k(
    profiles: pd.DataFrame, k_range: range = range(1, 11)
) -> list[float]:
    """Calcula inertia para distintos k (método del codo)."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(profiles)
        inertias.append(km.inertia_)
    return inertias


def segment_students(df: pd.DataFrame, k: int = 4) -> tuple[pd.DataFrame, KMeans]:
    """
    Segmenta estudiantes con K-Means.
    Retorna el DataFrame con columna 'Cluster' y el modelo entrenado.
    """
    profiles, _scaler = prepare_profiles(df)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(profiles)
    df_out = df.copy()
    df_out["Cluster"] = labels
    return df_out, kmeans


# ═══════════════════════════════════════════════════════════════════════════
#  PERFILES DE CLUSTERS
# ═══════════════════════════════════════════════════════════════════════════


def get_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen estadístico por cluster sobre variables clave."""
    key_cols = [
        "Age",
        "StudyTimeWeekly",
        "Absences",
        "GPA",
        "Tutoring",
        "Extracurricular",
        "Sports",
        "Music",
        "Volunteering",
    ]
    available = [c for c in key_cols if c in df.columns]
    summary = df.groupby("Cluster")[available].mean()
    summary["Estudiantes"] = df.groupby("Cluster").size()
    return summary


def _get_cluster_label(summary_row: pd.Series, global_medians: dict) -> str:
    """Genera una etiqueta descriptiva para un cluster."""
    gpa_high = summary_row["GPA"] >= global_medians["GPA"]
    abs_low = summary_row["Absences"] <= global_medians["Absences"]

    if gpa_high and abs_low:
        return "Alto Rendimiento · Buena Asistencia"
    elif gpa_high and not abs_low:
        return "Alto Rendimiento · Baja Asistencia"
    elif not gpa_high and abs_low:
        return "Bajo Rendimiento · Buena Asistencia"
    else:
        return "Bajo Rendimiento · Baja Asistencia"


def label_clusters(df: pd.DataFrame) -> dict[int, str]:
    """Asigna etiquetas descriptivas a cada cluster."""
    summary = get_cluster_summary(df)
    global_medians = {
        "GPA": df["GPA"].median(),
        "Absences": df["Absences"].median(),
    }
    return {
        cid: _get_cluster_label(summary.loc[cid], global_medians)
        for cid in summary.index
    }


# ═══════════════════════════════════════════════════════════════════════════
#  GRÁFICOS RADAR
# ═══════════════════════════════════════════════════════════════════════════


def compute_radar_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Datos normalizados (0-1) para gráfico radar por cluster.
    Convierte Absences en «Asistencia» (invertido: mayor = mejor).
    """
    radar_df = df[["Cluster"] + RADAR_RAW_COLS].copy()

    # 1. Invertimos las ausencias
    max_abs = radar_df["Absences"].max()
    radar_df["Asistencia"] = max_abs - radar_df["Absences"]
    radar_df = radar_df.drop(columns=["Absences"])

    # 2. DEFINIMOS LAS COLUMNAS A ESCALAR (todas menos 'Cluster')
    feature_cols = [c for c in radar_df.columns if c != "Cluster"]

    # 3. ESCALAMOS A TODOS LOS ESTUDIANTES PRIMERO (Este es el cambio clave)
    scaler = MinMaxScaler()
    radar_df[feature_cols] = scaler.fit_transform(radar_df[feature_cols])

    # 4. AGRUPAMOS Y PROMEDIAMOS LOS DATOS YA ESCALADOS
    means = radar_df.groupby("Cluster")[feature_cols].mean()

    # 5. Renombramos para el gráfico
    rename_map = {
        "GPA": "GPA",
        "StudyTimeWeekly": "Tiempo Estudio",
        "Asistencia": "Asistencia",
        "Tutoring": "Tutoría",
        "Extracurricular": "Extracurricular",
        "Sports": "Deportes",
    }
    means = means.rename(columns=rename_map)

    return means


def plot_radar_chart(
    radar_data: pd.DataFrame,
    cluster_id: int,
    label: str = "",
    color: str = "#1f77b4",
    ax=None,
) -> plt.Figure:
    """Genera gráfico radar para un cluster específico."""
    values = radar_data.loc[cluster_id].values.tolist()
    categories = radar_data.columns.tolist()
    n = len(categories)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    else:
        fig = ax.figure

    ax.fill(angles, values, color=color, alpha=0.25)
    ax.plot(angles, values, color=color, linewidth=2, marker="o", markersize=6)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=7, color="gray")
    title = f"Cluster {cluster_id}"
    if label:
        title += f"\n{label}"
    ax.set_title(title, size=13, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    return fig


def plot_all_radars(
    radar_data: pd.DataFrame, labels: dict[int, str] | None = None
) -> plt.Figure:
    """Genera todos los gráficos radar en una sola figura."""
    n_clusters = len(radar_data)
    cols = 2
    # Cálculo seguro de filas
    rows = int(np.ceil(n_clusters / cols))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, axes = plt.subplots(
        rows, cols, figsize=(12, 6 * rows), subplot_kw=dict(polar=True)
    )

    # Aseguramos que axes sea siempre un array 1D
    axes_flat = np.array(axes).flatten()

    # Iteramos sobre los clusters y dibujamos en sus respectivos ejes
    for i, cluster_id in enumerate(radar_data.index):
        color = colors[i % len(colors)]
        values = radar_data.loc[cluster_id].values.tolist()
        categories = radar_data.columns.tolist()
        n = len(categories)

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        ax = axes_flat[i]  # Tomamos el eje correcto

        ax.fill(angles, values, color=color, alpha=0.25)
        ax.plot(angles, values, color=color, linewidth=2, marker="o", markersize=6)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=7, color="gray")

        title = f"Cluster {cluster_id}"
        if labels and cluster_id in labels:
            title += f"\n{labels[cluster_id]}"
        ax.set_title(title, size=12, fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3)

    # Fuera del bucle principal, eliminamos los ejes que no se usaron
    for j in range(n_clusters, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    fig.suptitle(
        "Perfiles de Clusters – Gráfico Radar",
        size=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  RECOMENDACIONES POR CLUSTER
# ═══════════════════════════════════════════════════════════════════════════


def _unique_recs(recs: list[dict]) -> list[dict]:
    """Elimina recomendaciones duplicadas preservando orden."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in recs:
        key = r["nombre"]
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def get_recommendations_for_cluster(
    summary_row: pd.Series, global_medians: dict
) -> list[dict]:
    """Genera recomendaciones basadas en las características del cluster."""
    recs: list[dict] = []

    # GPA bajo → refuerzo
    if summary_row["GPA"] < global_medians["GPA"]:
        recs.append(MATERIAL_CATALOG["remedial_exercises"])
        recs.append(MATERIAL_CATALOG["online_courses"])
    else:
        recs.append(MATERIAL_CATALOG["advanced_materials"])

    # Altas ausencias → seguimiento
    if summary_row["Absences"] > global_medians["Absences"]:
        recs.append(MATERIAL_CATALOG["attendance_tracker"])

    # Bajo tiempo de estudio
    if summary_row["StudyTimeWeekly"] < global_medians.get("StudyTimeWeekly", 10):
        recs.append(MATERIAL_CATALOG["study_guides"])

    # Sin tutoría
    if summary_row.get("Tutoring", 0) < 0.5:
        recs.append(MATERIAL_CATALOG["tutoring"])

    # Sin extracurriculares
    if summary_row.get("Extracurricular", 0) < 0.5:
        recs.append(MATERIAL_CATALOG["extracurricular"])

    # Involucramiento familiar siempre útil
    recs.append(MATERIAL_CATALOG["family_engagement"])
    recs.append(MATERIAL_CATALOG["mentoring"])

    # GPA bajo + mayor edad → orientación vocacional
    if summary_row["GPA"] < global_medians["GPA"] and summary_row.get("Age", 16) > 17:
        recs.append(MATERIAL_CATALOG["vocational"])

    return _unique_recs(recs)


def get_all_cluster_recommendations(
    df: pd.DataFrame,
) -> dict[int, list[dict]]:
    """Genera recomendaciones para todos los clusters."""
    summary = get_cluster_summary(df)
    global_medians = {
        col: df[col].median() for col in NUMERICAL_COLS if col in df.columns
    }

    return {
        cid: get_recommendations_for_cluster(summary.loc[cid], global_medians)
        for cid in summary.index
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MODELO DE RECOMENDACIÓN ML  (KNN)
# ═══════════════════════════════════════════════════════════════════════════

KNN_FEATURE_COLS = [
    "StudyTimeWeekly",
    "Absences",
    "GPA",
    "Tutoring",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
]


def build_knn_model(
    df: pd.DataFrame, n_neighbors: int = 10
) -> tuple[NearestNeighbors, StandardScaler, list[str]]:
    """
    Construye un modelo KNN para encontrar estudiantes similares.
    Retorna (modelo, scaler, feature_cols).
    """
    available = [c for c in KNN_FEATURE_COLS if c in df.columns]
    features = df[available].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(features_scaled)
    return knn, scaler, available


def recommend_for_student(
    df: pd.DataFrame,
    student_idx: int,
    knn_model: NearestNeighbors,
    scaler: StandardScaler,
    feature_cols: list[str],
) -> dict:
    """
    Recomendaciones personalizadas para un estudiante.
    Busca vecinos con mejor GPA y sugiere lo que hacen diferente.
    """
    student = df.loc[student_idx]
    student_features = student[feature_cols].values.reshape(1, -1)
    student_scaled = scaler.transform(student_features)

    distances, indices = knn_model.kneighbors(student_scaled, n_neighbors=20)

    neighbors = df.iloc[indices[0]]
    better = neighbors[neighbors["GPA"] > student["GPA"]]

    if better.empty:
        better = neighbors.nlargest(5, "GPA")

    avg = better[feature_cols].mean()
    recs: list[dict] = []

    if avg["StudyTimeWeekly"] > student["StudyTimeWeekly"] * 1.15:
        recs.append(
            {
                **MATERIAL_CATALOG["study_guides"],
                "razon": (
                    f"Vecinos exitosos estudian ~{avg['StudyTimeWeekly']:.1f} hrs/sem "
                    f"vs tus {student['StudyTimeWeekly']:.1f} hrs."
                ),
            }
        )

    if student["Absences"] > avg["Absences"] * 1.15:
        recs.append(
            {
                **MATERIAL_CATALOG["attendance_tracker"],
                "razon": (
                    f"Tienes {student['Absences']:.0f} ausencias vs "
                    f"{avg['Absences']:.1f} de vecinos exitosos."
                ),
            }
        )

    if avg["Tutoring"] > 0.5 and student["Tutoring"] == 0:
        recs.append(
            {
                **MATERIAL_CATALOG["tutoring"],
                "razon": "La mayoría de vecinos exitosos utilizan tutorías.",
            }
        )

    if avg["Extracurricular"] > 0.5 and student["Extracurricular"] == 0:
        recs.append(
            {
                **MATERIAL_CATALOG["extracurricular"],
                "razon": "Vecinos exitosos participan en actividades extracurriculares.",
            }
        )

    if avg["Sports"] > 0.5 and student["Sports"] == 0:
        recs.append(
            {
                "nombre": "🏃 Actividades Deportivas",
                "descripcion": "Participar en deportes mejora concentración y rendimiento.",
                "tipo": "Desarrollo integral",
                "razon": "El deporte está correlacionado con mejor rendimiento en tus vecinos.",
            }
        )

    gpa_median = df["GPA"].median()
    if student["GPA"] < gpa_median:
        recs.append(
            {
                **MATERIAL_CATALOG["remedial_exercises"],
                "razon": (
                    f"Tu GPA ({student['GPA']:.2f}) está por debajo de "
                    f"la mediana ({gpa_median:.2f})."
                ),
            }
        )
    else:
        recs.append(
            {
                **MATERIAL_CATALOG["advanced_materials"],
                "razon": f"Tu GPA ({student['GPA']:.2f}) es bueno. ¡Puedes ir por más!",
            }
        )

    return {
        "estudiante": student.to_dict(),
        "vecinos_encontrados": len(better),
        "recomendaciones": _unique_recs(recs),
    }
