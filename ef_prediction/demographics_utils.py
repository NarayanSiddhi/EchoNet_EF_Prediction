"""11-D demographic vectors aligned with C3D-GAN / perfect-reconstruction conditioning."""

from __future__ import annotations

import ast
from typing import Tuple

import numpy as np
import pandas as pd

# Matches use_case_3_perfect_reconstruction/train_reconstruction.py
SEX_MAP = {"F": 0, "M": 1, "O": 0}
AGE_MAP = {"0-1": 0, "2-5": 1, "6-10": 2, "11-15": 3, "16-18": 4}
BMI_MAP = {"underweight": 0, "normal": 1, "overweight": 2, "obese": 3}


def compute_bmi_category(weight, height) -> str:
    if pd.isna(weight) or pd.isna(height) or height <= 0:
        return "normal"
    bmi_val = float(weight) / ((float(height) / 100.0) ** 2)
    if bmi_val < 18.5:
        return "underweight"
    if bmi_val < 25:
        return "normal"
    if bmi_val < 30:
        return "overweight"
    return "obese"


def indices_from_demo_vector(demo: np.ndarray) -> Tuple[int, int, int]:
    demo = np.asarray(demo, dtype=float).ravel()
    if demo.size == 11:
        sex = int(np.argmax(demo[0:2]))
        age_group = int(np.argmax(demo[2:7]))
        bmi_group = int(np.argmax(demo[7:11]))
        return sex, age_group, bmi_group
    if demo.size >= 14:
        sex = int(np.argmax(demo[0:2]))
        age_group = int(np.argmax(demo[2:10]))
        bmi_group = int(np.argmax(demo[10:14]))
        return sex, age_group, bmi_group
    return 0, 0, 0


def row_to_demo_vector(row: pd.Series) -> np.ndarray:
    """Return float32 vector [11]: sex(2) + age_bin(5) + bmi(4)."""
    if "demographics" in row.index and pd.notna(row.get("demographics", np.nan)):
        demo = row["demographics"]
        if isinstance(demo, str):
            demo = ast.literal_eval(demo)
        arr = np.asarray(demo, dtype=np.float32).ravel()
        if arr.size == 11:
            return arr.astype(np.float32, copy=False)

    sex_key = row["sex"] if "sex" in row.index else row.get("Sex", "F")
    if pd.isna(sex_key):
        sex_key = "F"
    sk = str(sex_key).strip().upper()
    s = SEX_MAP.get(sk, 0)

    age_bin = row.get("age_bin", "0-1")
    if pd.isna(age_bin):
        age_bin = "0-1"
    a = AGE_MAP.get(str(age_bin).strip(), 0)

    bmi_cat = row.get("bmi_category", None)
    if bmi_cat is None or (isinstance(bmi_cat, float) and np.isnan(bmi_cat)) or str(bmi_cat).strip() == "":
        bmi_cat = compute_bmi_category(row.get("weight"), row.get("height"))
    b = BMI_MAP.get(str(bmi_cat).strip().lower(), 1)

    v = np.zeros(11, dtype=np.float32)
    v[s] = 1.0
    v[2 + a] = 1.0
    v[7 + b] = 1.0
    return v
