"""데이터 로드·전처리·분할 공통 함수."""
from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder


# ── 파일 로드 ──────────────────────────────────────────────────────
def load_uploaded_file(uploaded_file, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Streamlit UploadedFile 객체에서 DataFrame을 읽어 반환."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    bio = io.BytesIO(raw)
    if name.endswith((".xlsx", ".xls")):
        if sheet_name:
            return pd.read_excel(bio, sheet_name=sheet_name)
        return pd.read_excel(bio)
    if name.endswith(".tsv"):
        return pd.read_csv(bio, sep="\t")
    return pd.read_csv(bio)


def list_excel_sheets(uploaded_file) -> list[str]:
    """업로드된 Excel 파일의 시트 목록을 반환 (CSV이면 빈 리스트)."""
    name = uploaded_file.name.lower()
    if not name.endswith((".xlsx", ".xls")):
        return []
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    bio = io.BytesIO(raw)
    xl = pd.ExcelFile(bio)
    return xl.sheet_names


# ── 컬럼 타입 감지 ─────────────────────────────────────────────────
def detect_column_types(df: pd.DataFrame, max_unique_for_cat: int = 20) -> dict:
    """수치형/범주형 컬럼을 자동 감지하여 dict 반환."""
    numeric, categorical, datetime_cols = [], [], []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) <= max_unique_for_cat and s.dtype.kind in "iu":
                categorical.append(col)
            else:
                numeric.append(col)
        else:
            categorical.append(col)
    return {"numeric": numeric, "categorical": categorical, "datetime": datetime_cols}


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """결측치 요약 DataFrame."""
    miss = df.isna().sum()
    rate = (miss / len(df) * 100).round(2)
    out = pd.DataFrame({
        "컬럼": miss.index,
        "결측 개수": miss.values,
        "결측 비율(%)": rate.values,
        "데이터 타입": [str(df[c].dtype) for c in miss.index],
    })
    return out.sort_values("결측 개수", ascending=False).reset_index(drop=True)


# ── 결측치 처리 ────────────────────────────────────────────────────
def impute_missing(df: pd.DataFrame, strategy: str = "median",
                   numeric_cols: Optional[list[str]] = None,
                   categorical_cols: Optional[list[str]] = None) -> pd.DataFrame:
    """결측치를 지정 전략으로 대체."""
    df = df.copy()
    if numeric_cols is None or categorical_cols is None:
        types = detect_column_types(df)
        numeric_cols = numeric_cols or types["numeric"]
        categorical_cols = categorical_cols or types["categorical"]

    if strategy == "drop":
        return df.dropna().reset_index(drop=True)

    if numeric_cols:
        if strategy in {"mean", "median", "most_frequent"}:
            imp = SimpleImputer(strategy=strategy)
            df[numeric_cols] = imp.fit_transform(df[numeric_cols])
        elif strategy == "knn":
            imp = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imp.fit_transform(df[numeric_cols])

    if categorical_cols:
        for col in categorical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0]
                                          if not df[col].mode().empty else "Unknown")

    return df


# ── 인코딩 ────────────────────────────────────────────────────────
def encode_categorical(df: pd.DataFrame, categorical_cols: list[str],
                        method: str = "label") -> tuple[pd.DataFrame, dict]:
    """범주형 컬럼을 Label / OneHot 인코딩."""
    df = df.copy()
    encoders: dict = {}
    if method == "onehot":
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
        encoders["method"] = "onehot"
        return df, encoders

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    encoders["method"] = "label"
    return df, encoders


# ── Train / Test 분할 ─────────────────────────────────────────────
def split_train_test(df: pd.DataFrame, target: str,
                      test_size: float = 0.2,
                      method: str = "random",
                      time_col: Optional[str] = None,
                      stratify: bool = True,
                      random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """랜덤 또는 시계열 분할."""
    if method == "time" and time_col:
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        cut = int(len(df_sorted) * (1 - test_size))
        train = df_sorted.iloc[:cut].copy()
        test = df_sorted.iloc[cut:].copy()
        return train, test

    from sklearn.model_selection import train_test_split
    strat = df[target] if stratify and df[target].nunique() < 30 else None
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ── 변수 분리 ──────────────────────────────────────────────────────
def get_xy(df: pd.DataFrame, target: str, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """X, y numpy array 반환."""
    return df[feature_cols].values, df[target].values
