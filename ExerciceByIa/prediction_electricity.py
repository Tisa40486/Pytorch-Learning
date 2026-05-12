"""
EXERCICE SIMPLE : Prédire la consommation électrique
======================================================
On va apprendre à prédire la consommation d'électricité
en fonction de la température dehors.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import t.pyplot as plt

# ============================================
# ÉTAPE 1 : Créer des données fictives
# ============================================

temperatures = np.array([5, 10, 15, 20, 25, 30, 35, 45]).reshape(-1, 1)

consommation = np.array([120, 110, 100, 80, 90, 100, 110, 120])

print("=" * 50)
print("DONNÉES D'ENTRAÎNEMENT")
print("=" * 50)
for temp, conso in zip(temperatures, consommation):
    print(f"Température: {temp[0]:5.1f}°C  →  Consommation: {conso:6.1f} kWh")

# ============================================
# ÉTAPE 2 : Créer et entraîner le modèle
# ============================================

model = LinearRegression()

model.fit(temperatures, consommation)

print("\n" + "=" * 50)
print("MODÈLE ENTRAÎNÉ")
print("=" * 50)
print(f"Coefficient (pente): {model.coef_[0]:.2f}")
print(f"Intercept (décalage): {model.intercept_:.2f}")
print(f"\nFormule: Consommation = {model.coef_[0]:.2f} × Température + {model.intercept_:.2f}")

# ============================================
# ÉTAPE 3 : Faire des prédictions
# ============================================

print("\n" + "=" * 50)
print("PRÉDICTIONS SUR LES DONNÉES D'ENTRAÎNEMENT")
print("=" * 50)

predictions = model.predict(temperatures)

for temp, vrai, pred in zip(temperatures, consommation, predictions):
    erreur = abs(vrai - pred)
    print(f"Température: {temp[0]:5.1f}°C  →  Vrai: {vrai:6.1f} kWh  |  Prédiction: {pred:6.1f} kWh  |  Erreur: {erreur:.1f}")

# ============================================
# ÉTAPE 4 : Prédire pour de nouvelles données
# ============================================

print("\n" + "=" * 50)
print("PRÉDICTIONS POUR NOUVELLES TEMPÉRATURES")
print("=" * 50)

nouvelles_temperatures = np.array([7, 12, 22, 32]).reshape(-1, 1)

for temp in nouvelles_temperatures:
    pred = model.predict(temp.reshape(-1, 1))[0]
    print(f"Température: {temp[0]:5.1f}°C  →  Prédiction: {pred:6.1f} kWh")

# ============================================
# ÉTAPE 5 : Visualiser les résultats
# ============================================

# Créer un graphique
plt.figure(figsize=(10, 6))

plt.scatter(temperatures, consommation, color='blue', s=100, label='Données réelles', zorder=3)

x_ligne = np.linspace(0, 35, 100).reshape(-1, 1)
y_ligne = model.predict(x_ligne)
plt.plot(x_ligne, y_ligne, color='red', linewidth=2, label='Modèle (droite prédite)')

# Ajouter des étiquettes
plt.xlabel('Température (°C)', fontsize=12)
plt.ylabel('Consommation électrique (kWh)', fontsize=12)
plt.title('Prédiction de la consommation électrique', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Sauvegarder l'image
plt.savefig('graphique_prediction.png')
print("\n✓ Graphique sauvegardé : graphique_prediction.png")

plt.show()

print("\n" + "=" * 50)
print("EXERCICE TERMINÉ !")
print("=" * 50)