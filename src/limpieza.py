import requests
import pandas as pd
import zipfile
import os
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/rabieelkharoua/students-performance-dataset"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw/")
FILE_NAME = os.path.join(RAW_DATA_PATH, "students_performance_dataset.zip")
DF_NAME = "Student_performance_data _.csv"

CLEAN_DATA_PATH = os.path.join(DATA_DIR, "clean/students_performance_clean.csv")


def download_data(url: str, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        response = requests.get(url)
        response.raise_for_status()
        if response.headers.get("Content-Type") == "application/zip":
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Archivo descargado exitosamente: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar el archivo: {e}")


def extract_zip(file_path: str, extract_to: str) -> None:
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Archivo extraído a: {extract_to}")


def one_hot_encode(df: pd.DataFrame, columns_list: list) -> pd.DataFrame:
    return pd.get_dummies(df, columns=columns_list, drop_first=True)


def standar_scaler(df: pd.DataFrame, columns_list: list) -> pd.DataFrame:
    scaler = StandardScaler()
    df[columns_list] = scaler.fit_transform(df[columns_list])
    return df


def save_clean_data(df: pd.DataFrame, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Archivo limpio guardado en: {file_path}")


def pipeline():
    download_data(DATA_URL, FILE_NAME)

    extract_zip(file_path=FILE_NAME, extract_to=RAW_DATA_PATH)

    raw = pd.read_csv(os.path.join(RAW_DATA_PATH, DF_NAME))

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

    numerical_cols = raw.columns.difference(categorical_cols)

    raw = one_hot_encode(raw, categorical_cols)
    raw = standar_scaler(raw, numerical_cols)

    save_clean_data(
        raw,
        CLEAN_DATA_PATH,
    )


def load_raw_data() -> pd.DataFrame:
    """Carga el CSV crudo sin transformaciones."""
    raw_path = os.path.join(RAW_DATA_PATH, DF_NAME)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"No se encontró el archivo: {raw_path}. "
            "Ejecuta pipeline() primero o coloca el CSV manualmente."
        )
    return pd.read_csv(raw_path)


# Mapeo de valores categóricos para display legible
CATEGORY_MAPS = {
    "Gender": {0: "Masculino", 1: "Femenino"},
    "Ethnicity": {0: "Caucásico", 1: "Afroamericano", 2: "Asiático", 3: "Otro"},
    "ParentalEducation": {
        0: "Ninguna",
        1: "Preparatoria",
        2: "Algo de Universidad",
        3: "Licenciatura",
        4: "Posgrado",
    },
    "Tutoring": {0: "No", 1: "Sí"},
    "ParentalSupport": {
        0: "Ninguno",
        1: "Bajo",
        2: "Moderado",
        3: "Alto",
        4: "Muy Alto",
    },
    "Extracurricular": {0: "No", 1: "Sí"},
    "Sports": {0: "No", 1: "Sí"},
    "Music": {0: "No", 1: "Sí"},
    "Volunteering": {0: "No", 1: "Sí"},
    "GradeClass": {
        0: "A (GPA ≥ 3.5)",
        1: "B (3.0–3.5)",
        2: "C (2.5–3.0)",
        3: "D (2.0–2.5)",
        4: "F (< 2.0)",
    },
}

CATEGORICAL_COLS = [
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

NUMERICAL_COLS = ["Age", "StudyTimeWeekly", "Absences", "GPA"]


def explore_data(df: pd.DataFrame) -> dict:
    """Retorna un diccionario con información exploratoria del DataFrame."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "nulls": df.isnull().sum().to_dict(),
        "total_nulls": int(df.isnull().sum().sum()),
        "describe": df[NUMERICAL_COLS].describe(),
        "categorical_value_counts": {
            col: df[col].value_counts().to_dict()
            for col in CATEGORICAL_COLS
            if col in df.columns
        },
        "correlation": df[NUMERICAL_COLS].corr(),
    }


if __name__ == "__main__":
    pipeline()
