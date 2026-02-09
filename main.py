import pandas as pd

df = pd.read_csv("df_venues_processed.csv", sep=";")

# print(df.head())

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
