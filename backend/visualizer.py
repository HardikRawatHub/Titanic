"""
visualizer.py
-------------
All chart-generation logic. Each function returns a base64-encoded
PNG string so it can travel cleanly over the FastAPI JSON response.
"""

import io
import base64
import warnings
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for threads)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Design system ──────────────────────────────────────────────────────────
PALETTE       = ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#f5a623"]
SURVIVAL_COLORS = {"Survived": "#2ecc71", "Did Not Survive": "#e74c3c"}
ACCENT        = "#e94560"
BG_COLOR      = "#0d1117"
GRID_COLOR    = "#21262d"
TEXT_COLOR    = "#c9d1d9"

def _apply_dark_theme(fig, ax_or_axes):
    """Apply a consistent dark theme to any chart."""
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    fig.patch.set_facecolor(BG_COLOR)
    for ax in np.array(axes).flatten():
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.7)


def _to_base64(fig) -> str:
    """Serialize a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Individual chart functions
# ═══════════════════════════════════════════════════════════════════════════

def age_histogram(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    age_data = df["age"].dropna()
    ax.hist(age_data, bins=30, color=ACCENT, edgecolor="#0d1117", alpha=0.85)
    ax.set_title("Age Distribution of Titanic Passengers", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of Passengers")
    ax.axvline(age_data.mean(), color="#f5a623", linestyle="--", linewidth=1.5,
               label=f"Mean age: {age_data.mean():.1f}")
    ax.axvline(age_data.median(), color="#4fc3f7", linestyle="--", linewidth=1.5,
               label=f"Median age: {age_data.median():.1f}")
    ax.legend(facecolor="#21262d", labelcolor=TEXT_COLOR, fontsize=9)
    _apply_dark_theme(fig, ax)
    return _to_base64(fig)


def survival_by_sex(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Count chart
    counts = df.groupby(["sex", "survived_label"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", ax=axes[0], color=[SURVIVAL_COLORS["Did Not Survive"],
                                                SURVIVAL_COLORS["Survived"]],
                edgecolor="none", width=0.6)
    axes[0].set_title("Survival Count by Sex", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Sex"); axes[0].set_ylabel("Count")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].legend(facecolor="#21262d", labelcolor=TEXT_COLOR, fontsize=8)

    # Rate chart
    rates = df.groupby("sex")["survived"].mean() * 100
    bars = axes[1].bar(rates.index, rates.values,
                       color=[ACCENT if s == "male" else "#4fc3f7" for s in rates.index],
                       edgecolor="none", width=0.5)
    for bar, val in zip(bars, rates.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", va="bottom", color=TEXT_COLOR, fontsize=10)
    axes[1].set_title("Survival Rate by Sex (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Sex"); axes[1].set_ylabel("Survival Rate (%)")
    axes[1].set_ylim(0, 100)

    _apply_dark_theme(fig, axes)
    fig.tight_layout(pad=2)
    return _to_base64(fig)


def fare_distribution(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fare = df["fare"].dropna()

    axes[0].hist(fare, bins=40, color="#4fc3f7", edgecolor="#0d1117", alpha=0.85)
    axes[0].set_title("Fare Distribution (Full Range)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Fare (£)"); axes[0].set_ylabel("Passengers")

    fare_clip = fare[fare < 200]
    axes[1].hist(fare_clip, bins=40, color=ACCENT, edgecolor="#0d1117", alpha=0.85)
    axes[1].set_title("Fare Distribution (< £200)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Fare (£)"); axes[1].set_ylabel("Passengers")
    axes[1].axvline(fare_clip.mean(), color="#f5a623", linestyle="--", linewidth=1.5,
                    label=f"Mean: £{fare_clip.mean():.2f}")
    axes[1].legend(facecolor="#21262d", labelcolor=TEXT_COLOR, fontsize=8)

    _apply_dark_theme(fig, axes)
    fig.tight_layout(pad=2)
    return _to_base64(fig)


def embarkation_counts(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    port_counts = df["embark_port"].value_counts().dropna()

    # Bar chart
    colors = ["#4fc3f7", ACCENT, "#f5a623"]
    bars = axes[0].bar(port_counts.index, port_counts.values, color=colors[:len(port_counts)],
                       edgecolor="none", width=0.5)
    for bar, val in zip(bars, port_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                     str(val), ha="center", va="bottom", color=TEXT_COLOR, fontsize=10)
    axes[0].set_title("Passengers by Embarkation Port", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Port"); axes[0].set_ylabel("Passengers")

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        port_counts.values, labels=port_counts.index,
        autopct="%1.1f%%", colors=colors[:len(port_counts)],
        startangle=140, pctdistance=0.82,
        wedgeprops={"edgecolor": BG_COLOR, "linewidth": 2}
    )
    for t in texts:    t.set_color(TEXT_COLOR)
    for t in autotexts: t.set_color(BG_COLOR); t.set_fontweight("bold")
    axes[1].set_title("Embarkation Port Share", fontsize=11, fontweight="bold")
    axes[1].title.set_color(TEXT_COLOR)

    _apply_dark_theme(fig, [axes[0]])  # pie ax handled separately
    axes[1].set_facecolor("#161b22")
    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout(pad=2)
    return _to_base64(fig)


def class_survival_heatmap(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot = df.pivot_table(values="survived", index="class_name",
                           columns="sex", aggfunc="mean") * 100
    pivot = pivot.reindex(["1st Class", "2nd Class", "3rd Class"])
    sns.heatmap(pivot, annot=True, fmt=".1f", ax=ax,
                cmap="RdYlGn", vmin=0, vmax=100,
                linewidths=0.5, linecolor=BG_COLOR,
                annot_kws={"size": 12, "weight": "bold"},
                cbar_kws={"label": "Survival Rate (%)"})
    ax.set_title("Survival Rate (%) by Class & Sex", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Sex"); ax.set_ylabel("Passenger Class")
    _apply_dark_theme(fig, ax)
    # Fix heatmap colorbar label color
    fig.axes[-1].yaxis.label.set_color(TEXT_COLOR)
    fig.axes[-1].tick_params(colors=TEXT_COLOR)
    return _to_base64(fig)


def family_size_survival(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    fam_surv = df.groupby("family_size")["survived"].mean() * 100
    fam_count = df.groupby("family_size").size()
    bars = ax.bar(fam_surv.index, fam_surv.values,
                  color=[ACCENT if v < 50 else "#2ecc71" for v in fam_surv.values],
                  edgecolor="none", width=0.6, alpha=0.85)
    for bar, val, cnt in zip(bars, fam_surv.values, fam_count.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%\n(n={cnt})", ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=8)
    ax.set_title("Survival Rate by Family Size", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Family Size (incl. self)"); ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 115)
    ax.axhline(50, color="#f5a623", linestyle="--", linewidth=1, alpha=0.6, label="50% line")
    ax.legend(facecolor="#21262d", labelcolor=TEXT_COLOR, fontsize=9)
    ax.set_xticks(fam_surv.index)
    _apply_dark_theme(fig, ax)
    return _to_base64(fig)


def age_survival_violin(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    df_clean = df.dropna(subset=["age"])
    palette = {"Survived": SURVIVAL_COLORS["Survived"],
               "Did Not Survive": SURVIVAL_COLORS["Did Not Survive"]}
    sns.violinplot(data=df_clean, x="survived_label", y="age",
                   palette=palette, ax=ax, inner="box", linewidth=0.8)
    ax.set_title("Age Distribution: Survived vs. Did Not Survive", fontsize=12,
                 fontweight="bold", pad=14)
    ax.set_xlabel("Outcome"); ax.set_ylabel("Age (years)")
    _apply_dark_theme(fig, ax)
    return _to_base64(fig)


def overview_dashboard(df: pd.DataFrame) -> str:
    """4-panel overview dashboard shown on first load."""
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("🚢  Titanic Dataset — Overview Dashboard",
                 fontsize=16, fontweight="bold", color=TEXT_COLOR, y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # 1. Survival overall
    sv = df["survived_label"].value_counts()
    ax1.pie(sv.values, labels=sv.index, autopct="%1.1f%%",
            colors=[SURVIVAL_COLORS[l] for l in sv.index],
            startangle=90, wedgeprops={"edgecolor": BG_COLOR, "linewidth": 2},
            pctdistance=0.75)
    ax1.set_title("Overall Survival", fontsize=10, fontweight="bold")
    ax1.title.set_color(TEXT_COLOR)
    ax1.set_facecolor("#161b22")

    # 2. Sex distribution
    sex_c = df["sex"].value_counts()
    ax2.bar(sex_c.index, sex_c.values, color=["#4fc3f7", ACCENT],
            edgecolor="none", width=0.5)
    ax2.set_title("Sex Distribution", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Passengers")

    # 3. Class distribution
    cls_c = df["class_name"].value_counts().sort_index()
    ax3.bar(cls_c.index, cls_c.values, color=["#f5a623", "#4fc3f7", ACCENT],
            edgecolor="none", width=0.5)
    ax3.set_title("Passenger Class", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Passengers")

    # 4. Age histogram
    ax4.hist(df["age"].dropna(), bins=25, color="#4fc3f7", edgecolor="#0d1117", alpha=0.85)
    ax4.set_title("Age Distribution", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Age"); ax4.set_ylabel("Count")

    # 5. Fare boxplot per class
    for i, (cls_name, grp) in enumerate(df.groupby("class_name")["fare"]):
        ax5.boxplot(grp.dropna(), positions=[i + 1], widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor=PALETTE[i % len(PALETTE)], color=TEXT_COLOR),
                    medianprops=dict(color="#f5a623", linewidth=2),
                    whiskerprops=dict(color=TEXT_COLOR),
                    capprops=dict(color=TEXT_COLOR),
                    flierprops=dict(marker="o", color=ACCENT, alpha=0.4, markersize=3))
    ax5.set_xticks([1, 2, 3])
    ax5.set_xticklabels(["1st", "2nd", "3rd"])
    ax5.set_title("Fare by Class", fontsize=10, fontweight="bold")
    ax5.set_xlabel("Class"); ax5.set_ylabel("Fare (£)")

    # 6. Survival rate by class
    surv_cls = df.groupby("class_name")["survived"].mean() * 100
    surv_cls = surv_cls.sort_index()
    bars = ax6.bar(surv_cls.index, surv_cls.values,
                   color=[ACCENT if v < 50 else "#2ecc71" for v in surv_cls.values],
                   edgecolor="none", width=0.5)
    for bar, val in zip(bars, surv_cls.values):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
    ax6.set_title("Survival Rate by Class", fontsize=10, fontweight="bold")
    ax6.set_ylabel("Survival Rate (%)")
    ax6.set_ylim(0, 100)

    _apply_dark_theme(fig, [ax2, ax3, ax4, ax5, ax6])
    return _to_base64(fig)


# ── Registry: keyword → function ───────────────────────────────────────────
CHART_REGISTRY = {
    "age_histogram":         age_histogram,
    "survival_by_sex":       survival_by_sex,
    "fare_distribution":     fare_distribution,
    "embarkation_counts":    embarkation_counts,
    "class_survival_heatmap": class_survival_heatmap,
    "family_size_survival":  family_size_survival,
    "age_survival_violin":   age_survival_violin,
    "overview_dashboard":    overview_dashboard,
}
