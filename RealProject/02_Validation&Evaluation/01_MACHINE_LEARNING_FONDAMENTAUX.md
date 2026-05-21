# 🤖 Machine Learning pour Débutants - Guide Complet

## Table des matières
1. [Concepts fondamentaux](#concepts-fondamentaux)
2. [Le Preprocessing](#1-le-preprocessing)
3. [Le Scaling](#2-le-scaling)
4. [Train/Test Split](#3-le-train-test-split)
5. [La Logistic Regression](#4-la-logistic-regression)
6. [Les Métriques](#5-les-métriques)
7. [La Matrice de Confusion](#6-la-matrice-de-confusion)
8. [Exemple Complet](#exemple-complet)

---

## Concepts Fondamentaux

### Qu'est-ce que le Machine Learning ?
Le ML c'est apprendre à un **modèle** à prédire quelque chose à partir de **données**.

**Processus :**
```
Données brutes → Preprocessing → Scaling → Entraînement → Prédictions
```

### Classification vs Régression
- **Classification** : Prédire une **catégorie** (ex: "hausse" ou "baisse")
- **Régression** : Prédire une **valeur numérique** (ex: le prix exact)

**Notre cas** : Classification binaire (2 classes : 0 ou 1)

---

# 1. Le Preprocessing

## Qu'est-ce que c'est ?

Le preprocessing c'est **nettoyer et préparer tes données** avant d'entraîner le modèle.

### Pourquoi c'est crucial ?
- Données avec des trous (NaN) → Le modèle ne peut pas apprendre
- Données corrompues → Mauvaises prédictions
- Données mal structurées → Erreurs

## Les étapes du preprocessing

### 1.1 Charger les données

```python
import pandas as pd

# Charger le CSV
data = pd.read_csv('trading_data_messy.csv')
df = pd.DataFrame(data)

# Afficher les infos
print(df.info())
print(df.head())  # Les 5 premières lignes
```

**Output :**
```
Date        Type: object
Ticker      Type: object
Open        Type: float64
High        Type: float64
Low         Type: float64
Close       Type: float64
Volume      Type: float64
Currency    Type: object
```

### 1.2 Vérifier les valeurs manquantes

```python
# Compter les NaN (valeurs manquantes)
print(df.isnull().sum())
```

**Output :**
```
Date       0
Ticker     0
Open       5
High       3
Low        2
Close      0
Volume     43    ← 43 valeurs manquantes !
Currency   0
```

**Visuel :**
```
Index | Open  | Volume    |
------|-------|-----------|
0     | 150.2 | 2345000   | ✅
1     | 151.9 | NaN       | ❌ Manquant !
2     | 152.3 | 2567000   | ✅
3     | NaN   | 2789000   | ❌ Manquant !
4     | 153.1 | 2634000   | ✅
```

### 1.3 Supprimer les lignes avec des NaN

```python
df_clean = df.dropna()

print(f"Avant: {len(df)} lignes")
print(f"Après: {len(df_clean)} lignes")
print(f"Supprimé: {len(df) - len(df_clean)} lignes")
```

**Output (cas réel) :**
```
Avant: 670 lignes
Après: 617 lignes
Supprimé: 53 lignes
```

**Pourquoi ?** Le modèle ne peut pas apprendre avec des valeurs manquantes. C'est mieux de perdre 53 lignes que d'avoir du bruit.

### 1.4 Sélectionner les colonnes pertinentes

```python
# Garder seulement ce dont on a besoin
X = df_clean[['open', 'high', 'low', 'volume']]  # Features (entrées)
y = (df_clean['close'] > df_clean['open']).astype(int)  # Target (sortie)

print(f"X shape: {X.shape}")  # (617, 4) = 617 lignes, 4 colonnes
print(f"y shape: {y.shape}")  # (617,) = 617 valeurs
```

**Explication du target :**
```python
df_clean['close'] > df_clean['open']
```
- Si `close > open` → **True** (la bourse a monté)
- Si `close <= open` → **False** (la bourse a baissé)

```python
.astype(int)
```
- Convertit **True en 1** et **False en 0**

**Exemple concret :**
```
| Index | Open  | Close | close > open | Target (y) |
|-------|-------|-------|-------------|-----------|
| 0     | 150.2 | 151.9 | True        | 1         |
| 1     | 151.9 | 153.7 | True        | 1         |
| 2     | 154.2 | 156.2 | True        | 1         |
| 3     | 157.9 | 158.4 | True        | 1         |
| 4     | 158.4 | 160.2 | True        | 1         |
```

---

# 2. Le Scaling

## Qu'est-ce que c'est ?

Le scaling c'est **normaliser les données** pour qu'elles soient sur la même échelle.

## Pourquoi c'est important ?

Tes features ont des **échelles très différentes** :

```
open   : 150 - 160
high   : 152 - 162
low    : 148 - 158
volume : 2000000 - 3000000    ← ÉNORME !
```

**Le problème :**
Le modèle voit `volume` comme beaucoup plus important car les nombres sont plus gros.
C'est **injuste** pour les autres colonnes.

## StandardScaler (ce qu'on utilise)

### Comment ça fonctionne ?

**Formule :**
```
X_scaled = (X - mean) / std_dev
```

**En français :**
- Soustrais la moyenne de chaque valeur
- Divise par l'écart-type

**Résultat :** Chaque colonne a une moyenne de **0** et un écart-type de **1**

### Exemple concret

**Avant scaling :**
```
Volume original: [2345000, 2567000, 2789000, 2634000]
Mean: 2583500
Std: 185500
```

**Calcul pour la première valeur :**
```
(2345000 - 2583500) / 185500 = -238500 / 185500 = -1.29
```

**Après scaling :**
```
Volume scaled: [-1.29, -0.08, 1.11, 0.27]
```

**Avant :**
```
2345000 (énorme)
```

**Après :**
```
-1.29 (comparable aux autres)
```

### Code complet

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# FIT (apprendre la moyenne et écart-type sur TRAIN)
X_train_scaled = scaler.fit_transform(X_train)

# TRANSFORM (appliquer sur TEST)
X_test_scaled = scaler.transform(X_test)

print("Avant scaling:")
print(X_train.head())
print("\nAprès scaling:")
print(X_train_scaled[:5])
```

**Output :**
```
Avant scaling:
     open    high     low     volume
0  150.25  152.80  149.50  2345000.0
1  151.95  154.30  151.20  2567000.0

Après scaling:
[[-1.85  -1.82  -1.79  -1.29]
 [-1.23  -1.20  -1.25  -0.08]
 [-0.85  -0.82  -0.91   1.11]
 [-0.42  -0.40  -0.50   0.27]]
```

### ⚠️ Règle importante

**FIT sur TRAIN, TRANSFORM sur TEST**

```python
# ✅ CORRECT
scaler.fit_transform(X_train)   # Apprendre sur train
scaler.transform(X_test)         # Utiliser sur test

# ❌ FAUX
scaler.fit_transform(X_test)    # Ne jamais "apprendre" sur test !
```

**Pourquoi ?** Si tu apprends sur test, tu "trichen" en donnant au modèle info sur les données de test.

---

# 3. Le Train/Test Split

## Qu'est-ce que c'est ?

Diviser tes données en **2 parties** :
- **Train (80%)** : pour entraîner le modèle
- **Test (20%)** : pour vérifier qu'il marche sur des données qu'il n'a jamais vues

## Pourquoi c'est crucial ?

Si tu entraînes ET testes sur les mêmes données, le modèle va **mémoriser** les réponses sans vraiment apprendre.

**Analogie :**
- Étudiant apprend un examen par cœur
- Passe un test identique → 100%
- Passe un test différent → 40%

C'est ça qui se passe si tu n'as pas de test set.

## Code

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% pour tester, 80% pour entraîner
    random_state=42,    # Pour reproductibilité
    shuffle=True        # Mélanger les données
)

print(f"Train set: {len(X_train)} lignes ({len(X_train)/len(X)*100:.0f}%)")
print(f"Test set: {len(X_test)} lignes ({len(X_test)/len(X)*100:.0f}%)")
```

**Output :**
```
Train set: 493 lignes (80%)
Test set: 124 lignes (20%)
```

## Visualisation

```
617 lignes totales
├─ 493 pour TRAIN (80%)
└─ 124 pour TEST (20%)
```

---

# 4. La Logistic Regression

## Qu'est-ce que c'est ?

C'est un **modèle de classification** qui prédit une probabilité entre 0 et 1.

## Comment ça fonctionne (simplifié)

Le modèle apprend une **relation linéaire** entre les features et la probabilité d'une classe.

**Formule simplifiée :**
```
Probabilité = 1 / (1 + e^(-z))
```

**Où z dépend de tes features :**
```
z = w1*open + w2*high + w3*low + w4*volume + bias
```

**Les w (poids) sont appris pendant l'entraînement.**

## Code

```python
from sklearn.linear_model import LogisticRegression

# Créer le modèle
model = LogisticRegression(random_state=42)

# Entraîner
model.fit(X_train_scaled, y_train)

# Prédire
y_pred = model.predict(X_test_scaled)

print(y_pred[:10])  # Les 10 premières prédictions
```

**Output :**
```
[1 1 1 0 1 1 1 1 1 0]
```

Chaque prédiction est **0 (baisse) ou 1 (hausse)**.

### Exemple concret d'une prédiction

**Données d'entrée (scalées) :**
```
[0.85, 0.92, 0.78, -1.2]
```

**Processus interne :**
```
z = 0.85*w1 + 0.92*w2 + 0.78*w3 + (-1.2)*w4 + bias
z = 0.85*2.1 + 0.92*1.8 + 0.78*1.5 + (-1.2)*0.9 + 0.3
z = 1.79 + 1.66 + 1.17 - 1.08 + 0.3
z = 3.84

Probabilité = 1 / (1 + e^(-3.84)) = 0.98  (très proche de 1)

Prédiction = 1 (hausse)
```

---

# 5. Les Métriques

## Qu'est-ce que c'est ?

Les métriques messurent **à quel point ton modèle est bon**.

## Les 4 principales

### 5.1 Accuracy (Précision globale)

**Définition :**
```
Accuracy = Nombre de bonnes prédictions / Total de prédictions
```

**Formule :**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Où :**
- **TP (True Positive)** : Prédit 1 et c'est 1 ✅
- **TN (True Negative)** : Prédit 0 et c'est 0 ✅
- **FP (False Positive)** : Prédit 1 mais c'est 0 ❌
- **FN (False Negative)** : Prédit 0 mais c'est 1 ❌

**Exemple concret :**
```
Vraies valeurs: [1, 0, 1, 1, 0, 1, 0, 0]
Prédictions:   [1, 0, 1, 0, 0, 1, 0, 1]
Correctes:     [✓, ✓, ✓, ✗, ✓, ✓, ✓, ✗]

6 bonnes sur 8 = 75% d'accuracy
```

**Dans notre cas :**
```
Accuracy = 54.84%  (mauvais)
```

### 5.2 Precision (Précision d'une classe)

**Définition :**
```
Precision = TP / (TP + FP)

"Quand je dis 1, j'ai raison combien de fois ?"
```

**Exemple :**
```
Je prédis "hausse" 100 fois
70 fois j'avais raison
30 fois j'avais tort

Precision = 70 / 100 = 70%
```

**Dans notre cas :**
```
Precision (classe 1) = 52.73%

Quand le modèle dit "hausse", il a raison 52% du temps.
```

### 5.3 Recall (Sensibilité)

**Définition :**
```
Recall = TP / (TP + FN)

"Je détecte combien de vraies positives ?"
```

**Exemple :**
```
Il y a vraiment 80 hausses dans les données
Mon modèle en détecte 75
Recall = 75 / 80 = 93.75%
```

**Dans notre cas :**
```
Recall (classe 1) = 93.55%

Le modèle détecte 93% des vraies hausses.
```

### 5.4 F1-Score

**Définition :**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)

"L'équilibre entre Precision et Recall"
```

**Pourquoi ?** Pour ne pas favoriser l'une ou l'autre.

**Exemple :**
```
Precision = 70%, Recall = 80%
F1 = 2 * (0.7 * 0.8) / (0.7 + 0.8) = 74.6%
```

**Dans notre cas :**
```
F1 = 67.44%  (moyen)
```

## Résumé des métriques

| Métrique | Notre cas | Interprétation |
|----------|-----------|----------------|
| Accuracy | 54.84% | À peine mieux que le hasard |
| Precision | 52.73% | 1 sur 2 fois où on dit "hausse", c'est faux |
| Recall | 93.55% | On détecte presque toutes les hausses |
| F1 | 67.44% | Équilibre moyen |

---

# 6. La Matrice de Confusion

## Qu'est-ce que c'est ?

Un **tableau** qui montre exactement où le modèle se trompe.

## Format

```
                 Prédit 0    Prédit 1
Vraiment 0         TN          FP
Vraiment 1         FN          TP
```

## Notre cas réel

```
Matrice de confusion:
[[10 52]
 [ 4 58]]
```

**Décodage :**
```
                Prédit 0 (baisse)    Prédit 1 (hausse)
Vraiment 0 (baisse)    10                  52
Vraiment 1 (hausse)     4                  58
```

### Analyse détaillée

| | Prédit 0 | Prédit 1 | Total |
|---|---|---|---|
| **Vraiment 0** | 10 ✅ | 52 ❌ | 62 |
| **Vraiment 1** | 4 ❌ | 58 ✅ | 62 |
| **Total** | 14 | 110 | 124 |

**Calculs :**
- **TP (True Positive)** = 58 : Dit "hausse" et c'est juste
- **TN (True Negative)** = 10 : Dit "baisse" et c'est juste
- **FP (False Positive)** = 52 : Dit "hausse" mais c'est une baisse ❌
- **FN (False Negative)** = 4 : Dit "baisse" mais c'est une hausse ❌

**Accuracy = (10 + 58) / 124 = 54.84% ✓**
**Precision = 58 / (58 + 52) = 52.73% ✓**
**Recall = 58 / (58 + 4) = 93.55% ✓**

### Visualisation

```
Le modèle prédit beaucoup de "1" (hausse)

Prédictions:
- 14 fois "baisse" (0)
- 110 fois "hausse" (1)  ← ÉNORME !

Résultat:
- Quand c'est vraiment une hausse (62 fois) : il en détecte 58 ✅
- Quand c'est vraiment une baisse (62 fois) : il en rate 52 (dit "hausse" au lieu de "baisse") ❌
```

---

# Exemple Complet

## Code intégral avec commentaires

```python
# ========== IMPORTS ==========
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ========== 1. PREPROCESSING ==========
print("=" * 50)
print("ÉTAPE 1: PREPROCESSING")
print("=" * 50)

# Charger les données
data = pd.read_csv('trading_data_messy.csv')
df = pd.DataFrame(data)
print(f"✓ Données chargées: {len(df)} lignes")

# Vérifier les données manquantes
missing = df.isnull().sum()
print(f"✓ Valeurs manquantes:\n{missing}")

# Supprimer les NaN
df_clean = df.dropna()
print(f"✓ Après nettoyage: {len(df_clean)} lignes")
print(f"  (Supprimé: {len(df) - len(df_clean)} lignes)")

# Sélectionner X et y
X = df_clean[['open', 'high', 'low', 'volume']]
y = (df_clean['close'] > df_clean['open']).astype(int)
print(f"✓ X shape: {X.shape}")
print(f"✓ y shape: {y.shape}")
print(f"✓ Distribution y: 0={sum(y==0)}, 1={sum(y==1)}")

# ========== 2. TRAIN/TEST SPLIT ==========
print("\n" + "=" * 50)
print("ÉTAPE 2: TRAIN/TEST SPLIT")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"✓ Train set: {len(X_train)} lignes ({len(X_train)/len(X)*100:.0f}%)")
print(f"✓ Test set: {len(X_test)} lignes ({len(X_test)/len(X)*100:.0f}%)")

# ========== 3. SCALING ==========
print("\n" + "=" * 50)
print("ÉTAPE 3: SCALING")
print("=" * 50)

print("Avant scaling (premières lignes):")
print(X_train.head())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nAprès scaling (premières lignes):")
print(X_train_scaled[:5])
print("✓ Mean ≈ 0, Std ≈ 1")

# ========== 4. ENTRAÎNEMENT ==========
print("\n" + "=" * 50)
print("ÉTAPE 4: ENTRAÎNEMENT")
print("=" * 50)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
print("✓ Modèle entraîné")

# ========== 5. PRÉDICTIONS ==========
print("\n" + "=" * 50)
print("ÉTAPE 5: PRÉDICTIONS")
print("=" * 50)

y_pred = model.predict(X_test_scaled)
print(f"✓ Prédictions effectuées: {len(y_pred)} valeurs")
print(f"  Premières prédictions: {y_pred[:10]}")

# ========== 6. MÉTRIQUES ==========
print("\n" + "=" * 50)
print("ÉTAPE 6: MÉTRIQUES")
print("=" * 50)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"✓ Accuracy:  {accuracy:.2%}")
print(f"✓ Precision: {precision:.2%}")
print(f"✓ Recall:    {recall:.2%}")
print(f"✓ F1-Score:  {f1:.2%}")

# ========== 7. MATRICE DE CONFUSION ==========
print("\n" + "=" * 50)
print("ÉTAPE 7: MATRICE DE CONFUSION")
print("=" * 50)

cm = confusion_matrix(y_test, y_pred)
print(f"Matrice:\n{cm}")

tn, fp, fn, tp = cm.ravel()
print(f"\n✓ TP (Vrai Positif):    {tp}")
print(f"✓ TN (Vrai Négatif):    {tn}")
print(f"✓ FP (Faux Positif):    {fp}")
print(f"✓ FN (Faux Négatif):    {fn}")

# ========== 8. RAPPORT DÉTAILLÉ ==========
print("\n" + "=" * 50)
print("ÉTAPE 8: RAPPORT DÉTAILLÉ")
print("=" * 50)

print(classification_report(y_test, y_pred, target_names=['Baisse', 'Hausse']))

# ========== 9. ANALYSE ==========
print("\n" + "=" * 50)
print("ÉTAPE 9: ANALYSE DU PROBLÈME")
print("=" * 50)

print(f"Prédictions 0 (baisse): {sum(y_pred == 0)}")
print(f"Prédictions 1 (hausse): {sum(y_pred == 1)}")
print(f"\nVraies 0 (baisse): {sum(y_test == 0)}")
print(f"Vraies 1 (hausse): {sum(y_test == 1)}")

print(f"\n⚠️ PROBLÈME DÉTECTÉ:")
print(f"   Le modèle prédit trop de '1' (hausse)")
print(f"   Il génère {fp} faux positifs")
print(f"   C'est {fp/sum(y_test == 0)*100:.0f}% des vraies baisses mal classées")
```

**Output complet :**
```
==================================================
ÉTAPE 1: PREPROCESSING
==================================================
✓ Données chargées: 670 lignes
✓ Valeurs manquantes:
date        0
ticker      0
open        5
high        3
low         2
close       0
volume     43
currency    0
dtype: int64
✓ Après nettoyage: 617 lignes
  (Supprimé: 53 lignes)
✓ X shape: (617, 4)
✓ y shape: (617,)
✓ Distribution y: 0=308, 1=309

==================================================
ÉTAPE 2: TRAIN/TEST SPLIT
==================================================
✓ Train set: 493 lignes (80%)
✓ Test set: 124 lignes (20%)

==================================================
ÉTAPE 3: SCALING
==================================================
Avant scaling (premières lignes):
     open    high     low     volume
0  150.25  152.80  149.50  2345000.0
1  151.95  154.30  151.20  2567000.0
2  154.20  156.75  153.90  2789000.0

Après scaling (premières lignes):
[[-1.85  -1.82  -1.79  -1.29]
 [-1.23  -1.20  -1.25  -0.08]
 [-0.85  -0.82  -0.91   1.11]]
✓ Mean ≈ 0, Std ≈ 1

==================================================
ÉTAPE 4: ENTRAÎNEMENT
==================================================
✓ Modèle entraîné

==================================================
ÉTAPE 5: PRÉDICTIONS
==================================================
✓ Prédictions effectuées: 124 valeurs
  Premières prédictions: [1 1 1 0 1 1 1 1 1 0]

==================================================
ÉTAPE 6: MÉTRIQUES
==================================================
✓ Accuracy:  54.84%
✓ Precision: 52.73%
✓ Recall:    93.55%
✓ F1-Score:  67.44%

==================================================
ÉTAPE 7: MATRICE DE CONFUSION
==================================================
Matrice:
[[10 52]
 [ 4 58]]

✓ TP (Vrai Positif):    58
✓ TN (Vrai Négatif):    10
✓ FP (Faux Positif):    52
✓ FN (Faux Négatif):    4

==================================================
ÉTAPE 8: RAPPORT DÉTAILLÉ
==================================================
              precision    recall  f1-score   support

       Baisse       0.71      0.16      0.26        62
       Hausse       0.53      0.94      0.67        62

    accuracy                           0.55       124
   macro avg       0.62      0.55      0.47       124
weighted avg       0.62      0.55      0.47       124

==================================================
ÉTAPE 9: ANALYSE DU PROBLÈME
==================================================
Prédictions 0 (baisse): 14
Prédictions 1 (hausse): 110

Vraies 0 (baisse): 62
Vraies 1 (hausse): 62

⚠️ PROBLÈME DÉTECTÉ:
   Le modèle prédit trop de '1' (hausse)
   Il génère 52 faux positifs
   C'est 84% des vraies baisses mal classées
```

---

## Résumé des concepts

| Concept | Rôle | Valeur type |
|---------|------|------------|
| **Preprocessing** | Nettoyer les données | Supprimer NaN |
| **Scaling** | Normaliser les features | Mean=0, Std=1 |
| **Train/Test** | Diviser les données | 80/20 |
| **Model** | Apprendre les patterns | LogisticRegression |
| **Accuracy** | Bonne prédictions / Total | 50-100% |
| **Precision** | TP / (TP + FP) | Important pour coût FP |
| **Recall** | TP / (TP + FN) | Important pour coût FN |
| **F1** | Équilibre Precision/Recall | 0-1 |

---

## Prochain étape

Maintenant que tu comprends les concepts :
1. **Améliorer le modèle** (réduire les faux positifs)
2. **Feature engineering** (ajouter de meilleures features)
3. **Essayer d'autres modèles** (Random Forest, SVM, etc.)

C'est bon ?
