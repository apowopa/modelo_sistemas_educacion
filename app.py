"""
Sistema de Recomendación de Materiales Educativos Personalizados
Streamlit App – Proyecto Modelado de Sistemas
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.limpieza import (
    load_raw_data,
    explore_data,
    CATEGORY_MAPS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
)
from src.analisis import (
    segment_students,
    prepare_profiles,
    find_optimal_k,
    get_cluster_summary,
    label_clusters,
    compute_radar_data,
    plot_radar_chart,
    plot_all_radars,
    get_all_cluster_recommendations,
    build_knn_model,
    recommend_for_student,
)

# ─── Configuración ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sistema de Recomendación Educativa",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state=240,
)

CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ─── Cache de datos ──────────────────────────────────────────────────────────


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_raw_data()


@st.cache_data
def run_segmentation(k: int) -> pd.DataFrame:
    df = load_data()
    df_seg, _model = segment_students(df, k=k)
    return df_seg


@st.cache_data
def get_elbow_inertias() -> list[float]:
    df = load_data()
    profiles, _ = prepare_profiles(df)
    return find_optimal_k(profiles)


@st.cache_data
def get_knn_artifacts(
    k: int,
) -> tuple:
    df_seg = run_segmentation(k)
    knn, scaler, feat_cols = build_knn_model(df_seg)
    return knn, scaler, feat_cols


# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("🎓")
st.sidebar.title(" Recomendación Educativa Personalizada")
st.sidebar.markdown(
    "Segmentación de estudiantes y recomendación "
    "de materiales educativos personalizados."
)
st.sidebar.markdown(
    "Desarrollado por:"
    "- Apolonio Cuevas Manuel"
    "- Sansores Arjona Alejandro]"
    "- Cauich Cauich Manuel]"
    "Proyecto de Sistemas de Apoyo a la toma de Descisiones"
    "Febrero 2026"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Fuente:** [Students Performance Dataset]"
    "(https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset)"
)
k_clusters = 2

# ─── Pestañas ────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Exploración de Datos",
        "🎯 Segmentación",
        "📋 Recomendaciones por Grupo",
        "🤖 Recomendación Individual",
    ]
)

# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 – Exploración de Datos
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("📊 Exploración del Dataset")

    df_raw = load_data()
    info = explore_data(df_raw)

    # Métricas rápidas
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Estudiantes", info["shape"][0])
    col_b.metric("Variables", info["shape"][1])
    col_c.metric("Valores nulos", info["total_nulls"])
    col_d.metric("GPA promedio", f"{df_raw['GPA'].mean():.2f}")

    st.subheader("Vista previa del dataset")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.subheader("Estadísticas descriptivas")
    st.dataframe(info["describe"], use_container_width=True)

    # Distribuciones numéricas
    st.subheader("Distribución de variables numéricas")
    fig_num, axes_num = plt.subplots(1, len(NUMERICAL_COLS), figsize=(16, 4))
    for i, col in enumerate(NUMERICAL_COLS):
        sns.histplot(df_raw[col], kde=True, ax=axes_num[i], color=CLUSTER_COLORS[i])
        axes_num[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig_num)

    # Correlación
    st.subheader("Matriz de correlación")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        info["correlation"],
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax_corr,
    )
    plt.tight_layout()
    st.pyplot(fig_corr)

    # Variables categóricas
    st.subheader("Variables categóricas")
    cat_col = st.selectbox("Selecciona una variable categórica:", CATEGORICAL_COLS)
    if cat_col:
        fig_cat, ax_cat = plt.subplots(figsize=(8, 4))
        counts = df_raw[cat_col].value_counts().sort_index()
        labels = [CATEGORY_MAPS.get(cat_col, {}).get(v, str(v)) for v in counts.index]
        ax_cat.bar(labels, counts.values, color="#1f77b4")
        ax_cat.set_title(f"Distribución de {cat_col}")
        ax_cat.set_ylabel("Frecuencia")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig_cat)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 – Segmentación
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("🎯 Segmentación de Estudiantes")

    # Método del codo
    st.subheader("Método del Codo")
    inertias = get_elbow_inertias()
    fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
    k_range = range(1, 11)
    ax_elbow.plot(k_range, inertias, marker="o", linewidth=2, color="#1f77b4")
    ax_elbow.axvline(
        x=k_clusters, color="red", linestyle="--", alpha=0.7, label=f"k={k_clusters}"
    )
    ax_elbow.set_xlabel("Número de clusters (k)")
    ax_elbow.set_ylabel("Inertia")
    ax_elbow.set_title("Método del codo para k óptimo")
    ax_elbow.set_xticks(list(k_range))
    ax_elbow.legend()
    ax_elbow.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_elbow)

    # Segmentar
    df_seg = run_segmentation(k_clusters)
    cluster_labels = label_clusters(df_seg)

    st.subheader(f"Resultados con k = {k_clusters}")

    # Resumen por cluster
    summary = get_cluster_summary(df_seg)
    st.dataframe(summary.round(2), use_container_width=True)

    # Etiquetas
    for cid, lbl in cluster_labels.items():
        n = int(summary.loc[cid, "Estudiantes"])
        st.markdown(f"- **Cluster {cid}** ({n} estudiantes): *{lbl}*")

    # Boxplots por cluster
    st.subheader("Distribución de variables por cluster")
    fig_box, axes_box = plt.subplots(2, 2, figsize=(14, 10))
    axes_box = axes_box.flatten()
    for i, feat in enumerate(["GPA", "Absences", "StudyTimeWeekly", "Age"]):
        sns.boxplot(
            x="Cluster",
            y=feat,
            data=df_seg,
            ax=axes_box[i],
            palette=CLUSTER_COLORS[:k_clusters],
            hue="Cluster",
            legend=False,
        )
        axes_box[i].set_title(f"{feat} por Cluster")
    plt.tight_layout()
    st.pyplot(fig_box)

    # Radar charts
    st.subheader("Gráficos Radar por Cluster")
    radar_data = compute_radar_data(df_seg)
    fig_radar = plot_all_radars(radar_data, labels=cluster_labels)
    st.pyplot(fig_radar)

    # Radar individual
    st.subheader("Radar individual")
    sel_cluster = st.selectbox(
        "Selecciona un cluster:",
        sorted(df_seg["Cluster"].unique()),
        format_func=lambda c: f"Cluster {c} – {cluster_labels.get(c, '')}",
    )
    color = CLUSTER_COLORS[sel_cluster % len(CLUSTER_COLORS)]
    fig_single = plot_radar_chart(
        radar_data, sel_cluster, label=cluster_labels.get(sel_cluster, ""), color=color
    )
    st.pyplot(fig_single)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 – Recomendaciones por Grupo
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("📋 Recomendaciones por Grupo")

    df_seg = run_segmentation(k_clusters)
    cluster_labels = label_clusters(df_seg)
    all_recs = get_all_cluster_recommendations(df_seg)
    summary = get_cluster_summary(df_seg)

    for cid in sorted(all_recs.keys()):
        lbl = cluster_labels.get(cid, "")
        n = int(summary.loc[cid, "Estudiantes"])
        with st.expander(f"🔹 Cluster {cid} – {lbl}  ({n} estudiantes)", expanded=True):
            # Mini perfil
            row = summary.loc[cid]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("GPA promedio", f"{row['GPA']:.2f}")
            mc2.metric("Ausencias", f"{row['Absences']:.1f}")
            mc3.metric("Hrs estudio/sem", f"{row['StudyTimeWeekly']:.1f}")
            mc4.metric("Edad promedio", f"{row['Age']:.1f}")

            st.markdown("**Materiales y estrategias recomendadas:**")
            for rec in all_recs[cid]:
                st.markdown(
                    f"- **{rec['nombre']}** · _{rec['tipo']}_\n  > {rec['descripcion']}"
                )


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 4 – Recomendación Individual (Modelo ML – KNN)
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("🤖 Recomendación Individual")
    st.markdown(
        "Selecciona un estudiante para obtener recomendaciones personalizadas "
        "basadas en sus vecinos más similares con mejor rendimiento (modelo KNN)."
    )

    df_seg = run_segmentation(k_clusters)
    knn, scaler, feat_cols = get_knn_artifacts(k_clusters)

    list_student_ids = df_seg["StudentID"].unique().tolist()

    # Selector de estudiante
    student_id = st.selectbox(
        "Selecciona un StudentID:",
        list_student_ids,
        format_func=lambda x: f"Student #{x}",
    )
    match = df_seg[df_seg["StudentID"] == student_id]

    if match.empty:
        st.warning(f"No se encontró un estudiante con StudentID = {student_id}.")
    else:
        idx = match.index[0]
        student = df_seg.loc[idx]

        # Perfil del estudiante
        st.subheader(f"Perfil del Estudiante #{student_id}")
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("GPA", f"{student['GPA']:.2f}")
        pc2.metric("Ausencias", f"{int(student['Absences'])}")
        pc3.metric("Hrs estudio/sem", f"{student['StudyTimeWeekly']:.1f}")
        pc4.metric("Cluster", f"{int(student['Cluster'])}")

        # Detalle categórico
        with st.expander("Detalles completos"):
            detail_cols = st.columns(3)
            items = [
                ("Edad", student["Age"]),
                (
                    "Género",
                    CATEGORY_MAPS["Gender"].get(student["Gender"], student["Gender"]),
                ),
                (
                    "Etnia",
                    CATEGORY_MAPS["Ethnicity"].get(
                        student["Ethnicity"], student["Ethnicity"]
                    ),
                ),
                (
                    "Educación Padres",
                    CATEGORY_MAPS["ParentalEducation"].get(
                        student["ParentalEducation"], student["ParentalEducation"]
                    ),
                ),
                (
                    "Tutoría",
                    CATEGORY_MAPS["Tutoring"].get(
                        student["Tutoring"], student["Tutoring"]
                    ),
                ),
                (
                    "Apoyo Parental",
                    CATEGORY_MAPS["ParentalSupport"].get(
                        student["ParentalSupport"], student["ParentalSupport"]
                    ),
                ),
                (
                    "Extracurricular",
                    CATEGORY_MAPS["Extracurricular"].get(
                        student["Extracurricular"], student["Extracurricular"]
                    ),
                ),
                (
                    "Deportes",
                    CATEGORY_MAPS["Sports"].get(student["Sports"], student["Sports"]),
                ),
                (
                    "Música",
                    CATEGORY_MAPS["Music"].get(student["Music"], student["Music"]),
                ),
                (
                    "Voluntariado",
                    CATEGORY_MAPS["Volunteering"].get(
                        student["Volunteering"], student["Volunteering"]
                    ),
                ),
                (
                    "Clase",
                    CATEGORY_MAPS["GradeClass"].get(
                        student["GradeClass"], student["GradeClass"]
                    ),
                ),
            ]
            for j, (name, val) in enumerate(items):
                detail_cols[j % 3].write(f"**{name}:** {val}")

        # Recomendaciones ML
        result = recommend_for_student(df_seg, idx, knn, scaler, feat_cols)

        st.subheader("Recomendaciones personalizadas")
        st.caption(
            f"Basadas en {result['vecinos_encontrados']} vecinos similares con mejor GPA."
        )

        if not result["recomendaciones"]:
            st.success("¡Excelente! Este estudiante ya tiene un perfil óptimo. 🎉")
        else:
            for rec in result["recomendaciones"]:
                with st.container():
                    st.markdown(f"### {rec['nombre']}")
                    st.markdown(f"*{rec['tipo']}* – {rec['descripcion']}")
                    if "razon" in rec:
                        st.info(f"💡 **¿Por qué?** {rec['razon']}")
                    st.markdown("---")
