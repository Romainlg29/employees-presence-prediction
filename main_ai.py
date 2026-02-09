import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

# Configuration du style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Chargement des données
df = pd.read_csv("df_venues_processed.csv", sep=";")

print(df.head())
print(df.columns)
print(df.info())

# Conversion de la date (format DD/MM/YYYY)
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df = df.sort_values("Date")

# ========================================
# 1. VISUALISATION DE LA SÉRIE TEMPORELLE
# ========================================

fig, axes = plt.subplots(4, 1, figsize=(15, 12))
fig.suptitle(
    "Analyse de la Série Temporelle des Réservations", fontsize=16, fontweight="bold"
)

# Tendance générale
axes[0].plot(df["Date"], df["Total_reservations"], linewidth=1.5, color="steelblue")
axes[0].set_title("Tendance Générale des Réservations", fontsize=12)
axes[0].set_ylabel("Réservations")
axes[0].grid(True, alpha=0.3)

# Moyenne mobile (7 jours et 30 jours)
df["MA_7"] = df["Total_reservations"].rolling(window=7, center=True).mean()
df["MA_30"] = df["Total_reservations"].rolling(window=30, center=True).mean()

axes[1].plot(
    df["Date"],
    df["Total_reservations"],
    alpha=0.3,
    label="Données brutes",
    color="lightgray",
)
axes[1].plot(
    df["Date"], df["MA_7"], linewidth=2, label="Moyenne mobile 7 jours", color="orange"
)
axes[1].plot(
    df["Date"], df["MA_30"], linewidth=2, label="Moyenne mobile 30 jours", color="red"
)
axes[1].set_title("Tendance avec Moyennes Mobiles", fontsize=12)
axes[1].set_ylabel("Réservations")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Saisonnalité hebdomadaire
weekly_pattern = df.groupby("jour_semaine")["Total_reservations"].mean().sort_index()
axes[2].bar(range(len(weekly_pattern)), weekly_pattern.values, color="teal", alpha=0.7)
axes[2].set_title("Saisonnalité Hebdomadaire (Moyenne par Jour)", fontsize=12)
axes[2].set_ylabel("Réservations Moyennes")
axes[2].set_xticks(range(len(weekly_pattern)))
axes[2].set_xticklabels(weekly_pattern.index, rotation=45, ha="right")
axes[2].grid(True, alpha=0.3, axis="y")

# Saisonnalité par semaine de l'année
weekly_season = df.groupby("Semaine")["Total_reservations"].mean()
axes[3].plot(
    weekly_season.index,
    weekly_season.values,
    marker="o",
    linewidth=2,
    markersize=4,
    color="purple",
)
axes[3].set_title("Saisonnalité par Semaine de l'Année", fontsize=12)
axes[3].set_xlabel("Semaine")
axes[3].set_ylabel("Réservations Moyennes")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("serie_temporelle.png", dpi=300, bbox_inches="tight")
print("\n✓ Graphique de série temporelle sauvegardé : serie_temporelle.png")

# ========================================
# 2. DÉTECTION DES ANOMALIES
# ========================================

fig, axes = plt.subplots(3, 1, figsize=(15, 10))
fig.suptitle("Détection des Anomalies", fontsize=16, fontweight="bold")

# Méthode 1: Z-Score
df["zscore"] = np.abs(stats.zscore(df["Total_reservations"]))
anomalies_zscore = df[df["zscore"] > 3]

axes[0].plot(
    df["Date"],
    df["Total_reservations"],
    linewidth=1,
    color="steelblue",
    label="Données normales",
)
axes[0].scatter(
    anomalies_zscore["Date"],
    anomalies_zscore["Total_reservations"],
    color="red",
    s=100,
    marker="o",
    label="Anomalies (Z-score > 3)",
    zorder=5,
)
axes[0].axhline(
    df["Total_reservations"].mean(),
    color="green",
    linestyle="--",
    alpha=0.5,
    label="Moyenne",
)
axes[0].set_title(
    f"Méthode Z-Score (Anomalies détectées: {len(anomalies_zscore)})", fontsize=12
)
axes[0].set_ylabel("Réservations")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Méthode 2: IQR (Interquartile Range)
Q1 = df["Total_reservations"].quantile(0.25)
Q3 = df["Total_reservations"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
anomalies_iqr = df[
    (df["Total_reservations"] < lower_bound) | (df["Total_reservations"] > upper_bound)
]

axes[1].plot(
    df["Date"],
    df["Total_reservations"],
    linewidth=1,
    color="steelblue",
    label="Données normales",
)
axes[1].scatter(
    anomalies_iqr["Date"],
    anomalies_iqr["Total_reservations"],
    color="orange",
    s=100,
    marker="s",
    label="Anomalies (IQR)",
    zorder=5,
)
axes[1].axhline(
    upper_bound, color="red", linestyle="--", alpha=0.5, label="Limites IQR"
)
axes[1].axhline(lower_bound, color="red", linestyle="--", alpha=0.5)
axes[1].set_title(
    f"Méthode IQR (Anomalies détectées: {len(anomalies_iqr)})", fontsize=12
)
axes[1].set_ylabel("Réservations")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Méthode 3: Écart par rapport à la moyenne mobile
df["deviation"] = np.abs(df["Total_reservations"] - df["MA_7"])
threshold = 2 * df["deviation"].std()
anomalies_ma = df[df["deviation"] > threshold]

axes[2].plot(
    df["Date"],
    df["Total_reservations"],
    linewidth=1,
    color="steelblue",
    label="Données réelles",
)
axes[2].plot(
    df["Date"],
    df["MA_7"],
    linewidth=2,
    color="green",
    alpha=0.7,
    label="Moyenne mobile 7j",
)
axes[2].scatter(
    anomalies_ma["Date"],
    anomalies_ma["Total_reservations"],
    color="purple",
    s=100,
    marker="^",
    label="Anomalies (écart > 2σ)",
    zorder=5,
)
axes[2].set_title(
    f"Écart à la Moyenne Mobile (Anomalies détectées: {len(anomalies_ma)})", fontsize=12
)
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Réservations")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("detection_anomalies.png", dpi=300, bbox_inches="tight")
print("✓ Graphique de détection d'anomalies sauvegardé : detection_anomalies.png")

# ========================================
# 3. ANALYSE DES CORRÉLATIONS
# ========================================

# Sélection des colonnes numériques pertinentes
numeric_cols = [
    "GLOBAL",
    "D1",
    "D2",
    "D3",
    "D4",
    "jour_feriÃ.",
    "pont.congÃ.",
    "holiday",
    "Temp",
    "pluie",
    "autre",
    "Greve_nationale",
    "SNCF",
    "prof_nationale",
    "Total_reservations",
]

correlation_matrix = df[numeric_cols].corr()

# Création de la heatmap
fig, axes = plt.subplots(2, 1, figsize=(14, 14))
fig.suptitle("Analyse des Corrélations", fontsize=16, fontweight="bold")

# Heatmap complète
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    ax=axes[0],
)
axes[0].set_title("Matrice de Corrélation Complète", fontsize=12, pad=10)

# Corrélations avec Total_reservations uniquement
correlations_with_target = (
    correlation_matrix["Total_reservations"]
    .drop("Total_reservations")
    .sort_values(ascending=False)
)
colors = ["green" if x > 0 else "red" for x in correlations_with_target.values]
axes[1].barh(
    range(len(correlations_with_target)),
    correlations_with_target.values,
    color=colors,
    alpha=0.7,
)
axes[1].set_yticks(range(len(correlations_with_target)))
axes[1].set_yticklabels(correlations_with_target.index)
axes[1].set_xlabel("Coefficient de Corrélation")
axes[1].set_title("Corrélations avec Total_reservations", fontsize=12)
axes[1].axvline(x=0, color="black", linestyle="-", linewidth=0.8)
axes[1].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("analyse_correlations.png", dpi=300, bbox_inches="tight")
print("✓ Graphique d'analyse des corrélations sauvegardé : analyse_correlations.png")

# ========================================
# STATISTIQUES RÉCAPITULATIVES
# ========================================

print("\n" + "=" * 60)
print("RÉSUMÉ DES ANALYSES")
print("=" * 60)

print(f"\n📊 STATISTIQUES DESCRIPTIVES:")
print(f"   • Moyenne des réservations: {df['Total_reservations'].mean():.2f}")
print(f"   • Médiane: {df['Total_reservations'].median():.2f}")
print(f"   • Écart-type: {df['Total_reservations'].std():.2f}")
print(f"   • Minimum: {df['Total_reservations'].min()}")
print(f"   • Maximum: {df['Total_reservations'].max()}")

print(f"\n🔍 ANOMALIES DÉTECTÉES:")
print(f"   • Méthode Z-Score (>3σ): {len(anomalies_zscore)} anomalies")
print(f"   • Méthode IQR: {len(anomalies_iqr)} anomalies")
print(f"   • Écart à la moyenne mobile: {len(anomalies_ma)} anomalies")

print(f"\n📈 CORRÉLATIONS FORTES (|r| > 0.5) avec Total_reservations:")
strong_correlations = correlations_with_target[abs(correlations_with_target) > 0.5]
if len(strong_correlations) > 0:
    for var, corr in strong_correlations.items():
        print(f"   • {var}: {corr:.3f}")
else:
    print("   • Aucune corrélation forte détectée")

print(f"\n🗓️ SAISONNALITÉ HEBDOMADAIRE:")
best_day = weekly_pattern.idxmax()
worst_day = weekly_pattern.idxmin()
print(f"   • Meilleur jour: {best_day} ({weekly_pattern.max():.1f} réservations)")
print(
    f"   • Jour le plus faible: {worst_day} ({weekly_pattern.min():.1f} réservations)"
)

print("\n✅ Toutes les analyses sont terminées!")
print("=" * 60)
