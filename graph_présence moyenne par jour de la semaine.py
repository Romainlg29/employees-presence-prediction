# 1. Présence moyenne par jour de la semaine (INDISPENSABLE)
# Graphique : Bar chart
# Axe X : jour_semaine
# Axe Y : moyenne de GLOBAL ou Total_reservations
# Montre clairement : forte présence lundi–vendredi chute nette samedi / dimanche

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration du style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Chargement des données
df = pd.read_csv('/Users/flaviechauvat/Documents/M2/PROJET/employees-presence-prediction/df_venues_processed.csv', sep=';')

# Conversion de la date
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Ordre des jours de la semaine
jours_ordre = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']

# Calculer la moyenne de présence par jour de la semaine
presence_par_jour = df.groupby('jour_semaine')['GLOBAL'].mean().reindex(jours_ordre)

# Créer le bar chart
fig, ax = plt.subplots(figsize=(12, 7))

# Couleurs différentes pour semaine vs weekend
couleurs = ['#2E86AB', '#2E86AB', '#2E86AB', '#2E86AB', '#2E86AB', '#E63946', '#E63946']

bars = ax.bar(range(len(presence_par_jour)), presence_par_jour.values, 
              color=couleurs, edgecolor='black', linewidth=1.5, alpha=0.8)

# Ajouter les valeurs au-dessus des barres
for i, (jour, valeur) in enumerate(presence_par_jour.items()):
    if pd.notna(valeur):
        ax.text(i, valeur + 10, f'{valeur:.0f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Personnalisation
ax.set_xlabel('Jour de la semaine', fontsize=14, fontweight='bold')
ax.set_ylabel('Présence moyenne (nombre d\'employés)', fontsize=14, fontweight='bold')
ax.set_title('Présence Moyenne par Jour de la Semaine', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(range(len(presence_par_jour)))
ax.set_xticklabels(presence_par_jour.index, fontsize=12)
ax.grid(axis='y', alpha=0.3)

# Ajouter une ligne de moyenne
moyenne_semaine = df[df['jour_semaine'].isin(jours_ordre[:5])]['GLOBAL'].mean()
ax.axhline(y=moyenne_semaine, color='orange', linestyle='--', 
           linewidth=2, label=f'Moyenne semaine: {moyenne_semaine:.0f}')

# Légende
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E86AB', edgecolor='black', label='Jours de semaine'),
    Patch(facecolor='#E63946', edgecolor='black', label='Weekend'),
    plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, 
               label=f'Moyenne semaine: {moyenne_semaine:.0f}')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig('presence_par_jour_semaine.png', dpi=300, bbox_inches='tight')
print("✓ Graphique sauvegardé : presence_par_jour_semaine.png")
plt.show()

# Afficher les statistiques
print("\n" + "="*60)
print("STATISTIQUES - PRÉSENCE PAR JOUR DE LA SEMAINE")
print("="*60)

for jour in jours_ordre:
    if jour in presence_par_jour.index and pd.notna(presence_par_jour[jour]):
        count = len(df[df['jour_semaine'] == jour])
        print(f"{jour.capitalize():12} : {presence_par_jour[jour]:6.1f} employés ({count} jours)")
    else:
        print(f"{jour.capitalize():12} : Pas de données")

# Calculer la différence semaine/weekend
if pd.notna(presence_par_jour.get('samedi')) or pd.notna(presence_par_jour.get('dimanche')):
    weekend_values = [v for j, v in presence_par_jour.items() if j in ['samedi', 'dimanche'] and pd.notna(v)]
    if weekend_values:
        moyenne_weekend = np.mean(weekend_values)
        print(f"\n{'Moyenne semaine':12} : {moyenne_semaine:.1f} employés")
        print(f"{'Moyenne weekend':12} : {moyenne_weekend:.1f} employés")
        print(f"{'Différence':12} : {moyenne_semaine - moyenne_weekend:.1f} employés ({((moyenne_semaine - moyenne_weekend)/moyenne_semaine)*100:.1f}% de baisse)")
else:
    print(f"\n{'Moyenne semaine':12} : {moyenne_semaine:.1f} employés")
    print("Pas de données de weekend disponibles")

print("="*60)


