# Les Fondations de Python, ML et PyTorch pour Débutants

## 📍 Partie 1 : Python (Les Bases)

### 1.1 Variables et Types

C'est comme des boîtes pour ranger des choses :

```python
# Une boîte qui contient un nombre entier
age = 25
print(age)  # Affiche : 25

# Une boîte qui contient du texte
nom = "Alice"
print(nom)  # Affiche : Alice

# Une boîte qui contient un nombre avec décimales
hauteur = 1.75
print(hauteur)  # Affiche : 1.75

# Une boîte qui contient vrai ou faux
est_vivant = True
print(est_vivant)  # Affiche : True
```

**Les types courants :**
- `int` : nombres entiers (5, -10, 0)
- `float` : nombres avec virgule (3.14, 1.75)
- `str` : du texte ("Bonjour", 'Chat')
- `bool` : vrai ou faux (True, False)

---

### 1.2 Les Listes (Collections de choses)

Une liste, c'est comme un panier où tu mets plusieurs choses :

```python
# Une liste de fruits
fruits = ["pomme", "banane", "orange"]
print(fruits)  # Affiche : ['pomme', 'banane', 'orange']

# Accéder à un élément (l'index commence à 0)
print(fruits[0])  # Affiche : pomme
print(fruits[1])  # Affiche : banane

# Ajouter un élément
fruits.append("raisin")
print(fruits)  # Affiche : ['pomme', 'banane', 'orange', 'raisin']

# Boucler dans une liste
for fruit in fruits:
    print(f"J'aime les {fruit}s")

# Accéder à la longueur
print(len(fruits))  # Affiche : 4
```

---

### 1.3 Les Dictionnaires (Clés et Valeurs)

C'est comme un annuaire téléphonique : tu cherches un nom, tu trouves le numéro :

```python
# Un dictionnaire
personne = {
    "nom": "Alice",
    "age": 25,
    "ville": "Paris"
}

# Accéder à une valeur avec sa clé
print(personne["nom"])      # Affiche : Alice
print(personne["age"])      # Affiche : 25

# Ajouter une nouvelle clé-valeur
personne["métier"] = "Ingénieur"
print(personne)
# Affiche : {'nom': 'Alice', 'age': 25, 'ville': 'Paris', 'métier': 'Ingénieur'}

# Parcourir le dictionnaire
for clé, valeur in personne.items():
    print(f"{clé}: {valeur}")
```

---

### 1.4 Les Boucles (Répéter des actions)

```python
# Boucle for : répéter un certain nombre de fois
for i in range(5):  # i va de 0 à 4
    print(f"Comptage : {i}")

# Boucle while : répéter tant qu'une condition est vraie
compteur = 0
while compteur < 3:
    print(f"Encore {compteur}")
    compteur += 1
```

---

### 1.5 Les Conditions (Si... alors...)

```python
age = 25

if age < 13:
    print("Tu es un enfant")
elif age < 18:
    print("Tu es un adolescent")
else:
    print("Tu es un adulte")
# Affiche : Tu es un adulte
```

---

### 1.6 Les Fonctions (Réutiliser du code)

C'est comme une recette : tu la définis une fois, et tu peux la réutiliser 1000 fois :

```python
# Définir une fonction
def additionner(a, b):
    """Cette fonction ajoute deux nombres"""
    resultat = a + b
    return resultat

# L'utiliser
print(additionner(5, 3))   # Affiche : 8
print(additionner(10, 20)) # Affiche : 30
```

---

### 1.7 Les Bibliothèques (Code prêt à l'emploi)

Python c'est puissant parce qu'il y a plein de code prêt à l'emploi :

```python
# Importer la bibliothèque math
import math

# L'utiliser
print(math.sqrt(16))  # Affiche : 4.0 (racine carrée)
print(math.pi)        # Affiche : 3.14159... (le nombre Pi)

# Ou importer une fonction spécifique
from math import sqrt
print(sqrt(25))       # Affiche : 5.0
```

---

## 📍 Partie 2 : Machine Learning (Les Concepts)

### 2.1 Qu'est-ce que le ML? C'est quoi l'idée?

**Sans ML (programmation classique) :**
Tu dis à l'ordinateur exactement quoi faire :
```python
def predire_prix_maison(taille_m2):
    # Je définis manuellement la formule
    prix = taille_m2 * 1000 + 50000
    return prix

print(predire_prix_maison(100))  # 150 000 euros
```

**Avec ML :**
Tu donnes à l'ordinateur plein d'exemples, et il apprend tout seul :
```
Exemple 1: 100 m² → 150 000 €
Exemple 2: 150 m² → 200 000 €
Exemple 3: 80 m² → 130 000 €
Exemple 4: 200 m² → 300 000 €
...
Ordinateur : "Ah d'accord! J'ai compris le pattern, la formule c'est : prix ≈ taille * 1000 + 50000"

Nouvelle maison : 120 m² ? → Ordinateur devine : 170 000 €
```

---

### 2.2 Les 3 Types de ML

#### 🔵 Supervised Learning (Apprentissage Supervisé)

Tu dis à l'ordinateur : "Voici les entrées ET les bonnes réponses"

**Exemple :** Prédire le prix d'une maison
- Entrée : taille, localisation, nombre de chambres
- Bonne réponse : prix réel de la maison
- L'ordinateur apprend : "Ah, si c'est grand + centre-ville → cher"

```python
# Données d'entraînement
donnees = [
    {"taille_m2": 100, "prix": 150000},
    {"taille_m2": 150, "prix": 200000},
    {"taille_m2": 80, "prix": 130000},
]

# L'algorithme apprend la relation entre taille et prix
```

#### 🟢 Unsupervised Learning (Apprentissage Non Supervisé)

Tu donnes juste les données, pas les réponses. L'ordinateur cherche des patterns :

**Exemple :** Grouper les clients similaires
- Données : achat historique, âge, localisation
- Pas de "bonne réponse" prédéfinie
- L'ordinateur trouve : "Ah, il y a 3 groupes de clients différents!"

```python
# Données sans label
clients = [
    {"age": 25, "achat_moyen": 50},
    {"age": 26, "achat_moyen": 55},
    {"age": 60, "achat_moyen": 200},
    {"age": 62, "achat_moyen": 210},
]

# L'algorithme dit : "Ces 2 premiers sont similaires, ces 2 derniers aussi"
```

#### 🔴 Reinforcement Learning (Apprentissage par Renforcement)

L'ordinateur essaie des actions, reçoit des récompenses ou des punitions :

**Exemple :** Un robot apprend à marcher
- Il essaie un mouvement → tombe (punition)
- Il essaie un autre → avance (récompense)
- Après 1000 essais → il marche bien!

---

### 2.3 Les Étapes du ML

1. **Collecter les données** : Avoir plein d'exemples
2. **Préparer les données** : Nettoyer, organiser
3. **Entraîner le modèle** : Laisser l'ordinateur apprendre
4. **Tester le modèle** : Vérifier si ça marche
5. **Utiliser le modèle** : Faire des prédictions sur nouvelles données

---

### 2.4 Les Concepts Clés

#### Modèle
C'est une formule mathématique complexe que l'ordinateur a apprise.

```
Avant : ???
Après entraînement : y = 0.8*x1 + 0.3*x2 - 0.1*x3 + 42
```

#### Overfitting (Surapprentissage)
L'ordinateur a trop bien mémorisé les exemples, pas assez généralisé :

```
Données d'entraînement : 100% de réussite
Nouvelles données : 50% de réussite ❌
```

#### Underfitting (Sous-apprentissage)
L'ordinateur n'a pas assez appris :

```
Données d'entraînement : 60% de réussite
Nouvelles données : 55% de réussite ❌
```

#### Bon apprentissage
```
Données d'entraînement : 95% de réussite
Nouvelles données : 93% de réussite ✅
```

---

## 📍 Partie 3 : PyTorch (Outil Magique pour ML)

### 3.1 Qu'est-ce que PyTorch?

C'est une bibliothèque Python pour faire du machine learning plus facilement.

Sans PyTorch : tu dois coder toutes les formules mathématiques toi-même (très compliqué)
Avec PyTorch : il fait beaucoup de trucs pour toi automatiquement

---

### 3.2 Les Tensors (Données de PyTorch)

Un **tensor** c'est juste un array multidimensionnel. C'est comme une boîte avec des compartiments :

```python
import torch

# Un nombre simple (tensor 0D)
x = torch.tensor(5)
print(x)  # tensor(5)

# Une liste (tensor 1D) - comme un vecteur
nombres = torch.tensor([1, 2, 3, 4])
print(nombres)  # tensor([1, 2, 3, 4])

# Une grille (tensor 2D) - comme une matrice
grille = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
print(grille)
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# Vérifier la forme
print(grille.shape)  # torch.Size([2, 3]) = 2 lignes, 3 colonnes

# Accéder à un élément
print(grille[0, 1])  # Affiche : 2 (ligne 0, colonne 1)
```

**Pourquoi des Tensors?** Parce que PyTorch peut les manipuler super vite sur GPU (carte graphique).

---

### 3.3 Opérations Basiques sur Tensors

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Addition
print(a + b)  # tensor([5, 7, 9])

# Multiplication
print(a * b)  # tensor([4, 10, 18])

# Multiplication par un nombre
print(a * 2)  # tensor([2, 4, 6])

# Opérations plus complexes
print(torch.sqrt(torch.tensor([4.0, 9.0])))  # tensor([2., 3.])

# Calcul de moyenne
print(a.mean())  # tensor(2.)

# Forme (shape)
print(a.shape)  # torch.Size([3])
```

---

### 3.4 Créer un Modèle Simple

Un modèle c'est juste une fonction que PyTorch va optimiser :

```python
import torch
import torch.nn as nn

# Définir un modèle simple
class MonModele(nn.Module):
    def __init__(self):
        super().__init__()
        # Une couche linéaire : entre 10 entrées, 5 sorties
        self.fc1 = nn.Linear(10, 5)
        # Une deuxième couche : entre 5 entrées, 1 sortie
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        # Passer les données dans la première couche
        x = self.fc1(x)
        # Ajouter une activation (rend le modèle non-linéaire)
        x = torch.relu(x)
        # Passer dans la deuxième couche
        x = self.fc2(x)
        return x

# Créer une instance du modèle
modele = MonModele()

# Données d'entrée aléatoires (batch de 3 exemples, 10 caractéristiques chacun)
donnees = torch.randn(3, 10)

# Faire une prédiction
predictions = modele(donnees)
print(predictions.shape)  # torch.Size([3, 1]) : 3 prédictions
print(predictions)
```

**Explication :**
- `Linear(10, 5)` : prend 10 nombres, sort 5 nombres
- `relu` : fonction d'activation qui rend le modèle "intelligent"
- `forward` : c'est la formule du modèle

---

### 3.5 Entraîner un Modèle (L'apprentissage)

C'est le cœur du ML : faire apprendre le modèle :

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Créer le modèle
modele = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# 2. Créer une fonction de perte (mesure d'erreur)
loss_fn = nn.MSELoss()  # Mean Squared Error

# 3. Créer un optimiseur (la formule pour améliorer le modèle)
optimizer = optim.SGD(modele.parameters(), lr=0.01)

# 4. Données d'entraînement
X_train = torch.randn(100, 10)  # 100 exemples, 10 caractéristiques
y_train = torch.randn(100, 1)   # 100 réponses

# 5. Boucle d'entraînement
for epoch in range(10):  # 10 tours d'entraînement
    
    # Forward pass : faire une prédiction
    predictions = modele(X_train)
    
    # Calculer l'erreur
    loss = loss_fn(predictions, y_train)
    
    # Backward pass : calculer les gradients
    optimizer.zero_grad()  # Reset les gradients
    loss.backward()        # Calculer comment changer les poids
    
    # Update les poids
    optimizer.step()
    
    # Afficher la progression
    print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")
```

**Explication du cycle :**
1. **Forward pass** : données → prédictions
2. **Calculer l'erreur** : prédictions vs vraies réponses
3. **Backward pass** : calculer comment s'améliorer
4. **Update** : changer les poids du modèle

C'est répété des centaines/milliers de fois jusqu'à ce que le modèle s'améliore.

---

### 3.6 Exemple Complet : Prédire des Maisons

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Données synthétiques
# Entrée : taille (m²), nombre de chambres
# Sortie : prix en milliers d'euros
X = torch.tensor([
    [100, 3],
    [150, 4],
    [80, 2],
    [200, 5],
    [120, 3],
    [180, 4],
], dtype=torch.float32)

y = torch.tensor([
    [150],
    [200],
    [130],
    [300],
    [170],
    [250],
], dtype=torch.float32)

# Modèle
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# Loss et Optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entraînement
for epoch in range(100):
    # Forward
    pred = model(X)
    loss = loss_fn(pred, y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

# Test : prédire le prix d'une maison de 130m² avec 3 chambres
nouvelle_maison = torch.tensor([[130, 3]], dtype=torch.float32)
prix_predit = model(nouvelle_maison)
print(f"\nMaison 130m² + 3 chambres → Prix prédit: {prix_predit.item():.1f}k€")
```

---

## 📍 Résumé Visuel

### Le Workflow ML

```
┌─────────────────┐
│ Données Brutes. │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Préparer Données    │ (nettoyage, normalisation)
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Créer un Modèle     │ (définir la structure)
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Entraîner Modèle    │ (loop d'apprentissage)
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Évaluer Modèle      │ (tester sur nouvelles données)
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Faire Prédictions   │ (utiliser en production)
└──────────────────────┘
```

---

## 🎯 Les Points Clés à Retenir

| Concept | Explication Simple |
|---------|-------------------|
| **Tensor** | Une boîte de nombres organisée en dimensions |
| **Modèle** | Une formule mathématique que l'ordinateur apprend |
| **Loss** | L'erreur du modèle (on veut la minimiser) |
| **Optimizer** | La stratégie pour améliorer le modèle |
| **Epoch** | Un tour d'entraînement complet |
| **Forward Pass** | Données → Prédictions |
| **Backward Pass** | Calculer comment s'améliorer |
| **Gradient** | Direction et force pour changer les poids |

---

## 💡 Conseils Pratiques

1. **Commencer simple** : avant de faire un réseau complexe, essaie avec peu de données et petit modèle
2. **Checker les formes** : `print(tensor.shape)` est ton meilleur ami
3. **Normaliser les données** : mettre toutes les valeurs entre 0 et 1 aide beaucoup
4. **Visualiser les résultats** : utilise matplotlib pour voir ce qui se passe
5. **Lire les erreurs** : elles disent ce qui va pas

---

## 🚀 Prochaines Étapes

1. Installer PyTorch : `pip install torch`
2. Essayer les exemples ici
3. Faire des petits projets (prédire prix, classer images, etc.)
4. Progressivement augmenter la complexité
5. Apprendre les architectures populaires (CNN, LSTM, Transformers)

Bonne chance! 🎯