"""
data_loader.py
--------------
Downloads and preprocesses the Titanic dataset from the Seaborn
built-in datasets. Returns a clean, enriched DataFrame that the
agent tools can query.
"""

import seaborn as sns
import pandas as pd
import numpy as np


def load_titanic() -> pd.DataFrame:
    """
    Load the Titanic dataset (bundled with seaborn) and return an
    enriched, analysis-ready DataFrame.
    """
    df = sns.load_dataset("titanic")

    # ── Rename for clarity ──────────────────────────────────────
    df = df.rename(columns={
        "pclass":   "ticket_class",
        "sibsp":    "siblings_spouses_aboard",
        "parch":    "parents_children_aboard",
        "embarked": "embark_port_code",
        "class":    "class_label",
    })

    # ── Derived columns ─────────────────────────────────────────
    df["family_size"]  = df["siblings_spouses_aboard"] + df["parents_children_aboard"] + 1
    df["is_alone"]     = (df["family_size"] == 1).astype(int)
    df["fare_per_person"] = (df["fare"] / df["family_size"]).round(2)

    # Age buckets
    bins   = [0, 12, 18, 35, 60, 120]
    labels = ["Child (0-12)", "Teen (13-18)", "Adult (19-35)",
              "Middle-aged (36-60)", "Senior (61+)"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

    # Full embarkation port name
    port_map = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    df["embark_port"] = df["embark_port_code"].map(port_map)

    # Survival label
    df["survived_label"] = df["survived"].map({1: "Survived", 0: "Did Not Survive"})

    # Ticket class label
    df["class_name"] = df["ticket_class"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})

    return df


# Singleton – loaded once on module import
TITANIC_DF: pd.DataFrame = load_titanic()


def get_df() -> pd.DataFrame:
    return TITANIC_DF


def get_summary_stats() -> dict:
    """Return high-level summary stats useful for the agent's context."""
    df = TITANIC_DF
    return {
        "total_passengers":   len(df),
        "survivors":          int(df["survived"].sum()),
        "survival_rate_pct":  round(df["survived"].mean() * 100, 2),
        "male_count":         int((df["sex"] == "male").sum()),
        "female_count":       int((df["sex"] == "female").sum()),
        "avg_age":            round(df["age"].mean(), 2),
        "avg_fare":           round(df["fare"].mean(), 2),
        "median_fare":        round(df["fare"].median(), 2),
        "class_distribution": df["ticket_class"].value_counts().to_dict(),
        "port_distribution":  df["embark_port"].value_counts().dropna().to_dict(),
        "columns":            list(df.columns),
    }
