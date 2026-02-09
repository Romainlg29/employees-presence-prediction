import pandas as pd

df = pd.read_csv("df_venues_processed.csv", sep=";")

print(df.info())
print(df.describe())

print(df.columns)

# Plot D1, D2, D3, D4 by "jour_semaine"
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="jour_semaine", y="D1", label="D1")
sns.lineplot(data=df, x="jour_semaine", y="D2", label="D2")
sns.lineplot(data=df, x="jour_semaine", y="D3", label="D3")
sns.lineplot(data=df, x="jour_semaine", y="D4", label="D4")
plt.title("D1, D2, D3, D4 par jour de la semaine")
plt.xlabel("Jour de la semaine")
plt.ylabel("Valeur")
plt.legend()
plt.show()

# Plot "GLOBAL" by "jour_semaine"
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="jour_semaine", y="GLOBAL", label="GLOBAL")
plt.title("Présence par jour de la semaine")
plt.xlabel("Jour de la semaine")
plt.ylabel("Présence")
plt.legend()
plt.show()

# Plot "GLOBAL" by "Date"
plt.figure(figsize=(12, 6))

# Create separate masks
pont_mask = df["pont.congÃ."] == 1
holiday_mask = df["holiday"] == 1
ferie_mask = df["jour_feriÃ."] == 1
normal_mask = ~(pont_mask | holiday_mask | ferie_mask)

# Plot all points
plt.plot(df["Date"], df["GLOBAL"], color="blue", linewidth=1, alpha=0.5)

# Highlight jour_ferié points in red
plt.scatter(
    df[ferie_mask]["Date"],
    df[ferie_mask]["GLOBAL"],
    color="yellow",
    s=100,
    zorder=6,
    edgecolors="black",
    linewidth=1.5,
    label="Jour Férié",
)

# Highlight pont.congé points in yellow
plt.scatter(
    df[pont_mask]["Date"],
    df[pont_mask]["GLOBAL"],
    color="yellow",
    s=100,
    zorder=5,
    edgecolors="black",
    linewidth=1.5,
    label="Pont/Congé",
)

# Highlight holiday points in gray
plt.scatter(
    df[holiday_mask]["Date"],
    df[holiday_mask]["GLOBAL"],
    color="gray",
    s=100,
    zorder=5,
    edgecolors="black",
    linewidth=1.5,
    label="Holiday",
)

# Plot normal points in blue
plt.scatter(
    df[normal_mask]["Date"],
    df[normal_mask]["GLOBAL"],
    color="blue",
    s=30,
    zorder=4,
    alpha=0.6,
)

plt.title("Présence par date")
plt.xlabel("Date")
plt.ylabel("Présence")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Calculate max values for normalization
max_D1 = df["D1"].max()
max_D2 = df["D2"].max()
max_D3 = df["D3"].max()
max_D4 = df["D4"].max()

print(f"Max employees - D1: {max_D1}, D2: {max_D2}, D3: {max_D3}, D4: {max_D4}")

# Normalize the values
df["D1_norm"] = df["D1"] / max_D1
df["D2_norm"] = df["D2"] / max_D2
df["D3_norm"] = df["D3"] / max_D3
df["D4_norm"] = df["D4"] / max_D4

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="jour_semaine", y="D1_norm", label="D1 (normalized)")
sns.lineplot(data=df, x="jour_semaine", y="D2_norm", label="D2 (normalized)")
sns.lineplot(data=df, x="jour_semaine", y="D3_norm", label="D3 (normalized)")
sns.lineplot(data=df, x="jour_semaine", y="D4_norm", label="D4 (normalized)")
plt.title("D1, D2, D3, D4 normalisés par le max d'employés - par jour de la semaine")
plt.xlabel("Jour de la semaine")
plt.ylabel("Taux d'occupation (0-1)")
plt.legend()
plt.show()
