# Cours Complet : Matplotlib - La Visualisation de Données en Python

## Table des matières
1. [Introduction](#introduction)
2. [Installation et Configuration](#installation-et-configuration)
3. [Concepts Fondamentaux](#concepts-fondamentaux)
4. [Les Graphiques de Base](#les-graphiques-de-base)
5. [Personnalisation Avancée](#personnalisation-avancée)
6. [Gestion des Figures et Sous-graphiques](#gestion-des-figures-et-sous-graphiques)
7. [Graphiques Spécialisés](#graphiques-spécialisés)
8. [Bonnes Pratiques](#bonnes-pratiques)
9. [Intégration avec NumPy et Pandas](#intégration-avec-numpy-et-pandas)

---

## Introduction

**Matplotlib** est la bibliothèque Python la plus populaire pour créer des visualisations statiques, animées et interactives. Elle offre un contrôle granulaire sur chaque élément du graphique.

### Pourquoi Matplotlib ?
- ✅ Contrôle total sur l'apparence
- ✅ Production de qualité publication
- ✅ Compatible avec Jupyter notebooks
- ✅ Extensible et personnalisable
- ✅ Intégration facile avec NumPy et Pandas

---

## Installation et Configuration

### Installation
```bash
pip install matplotlib
pip install numpy pandas  # recommandé
```

### Import basique
```python
import matplotlib.pyplot as plt
import numpy as np
```

### Vérifier l'installation
```python
import matplotlib
print(matplotlib.__version__)
```

---

## Concepts Fondamentaux

### Architecture de Matplotlib

Matplotlib utilise une hiérarchie d'objets :

```
Figure (fenêtre principale)
└── Axes (zone de tracé)
    ├── Courbes, barres, points, etc.
    ├── Axes X et Y
    ├── Grille
    └── Légende, titre, labels
```

### Les deux interfaces

#### 1. Interface pyplot (simple, rapide)
```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.show()
```

#### 2. Interface orientée objet (recommandée pour les projets complexes)
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.show()
```

**Recommandation** : Utilisez l'interface orientée objet pour plus de contrôle et de maintenabilité.

---

## Les Graphiques de Base

### 1. Graphique en courbes (Line plot)

```python
import matplotlib.pyplot as plt
import numpy as np

# Données
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Créer la figure et les axes
fig, ax = plt.subplots(figsize=(10, 6))

# Tracer la courbe
ax.plot(x, y, color='blue', linewidth=2, label='sin(x)')

# Personnalisation
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Graphique en courbes')
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()
```

**Paramètres courants** :
- `color` : couleur de la ligne
- `linewidth` ou `lw` : épaisseur
- `linestyle` ou `ls` : `'-'`, `'--'`, `'-.'`, `':'`
- `marker` : `'o'`, `'s'`, `'^'`, etc.
- `label` : légende

### 2. Graphique en barres (Bar plot)

```python
fig, ax = plt.subplots()

catégories = ['A', 'B', 'C', 'D']
valeurs = [10, 24, 36, 18]

ax.bar(catégories, valeurs, color='skyblue', edgecolor='navy')

ax.set_ylabel('Valeurs')
ax.set_title('Graphique en barres')
ax.grid(axis='y', alpha=0.3)

plt.show()
```

**Variantes** :
```python
# Barres horizontales
ax.barh(catégories, valeurs)

# Barres groupées
x = np.arange(len(catégories))
width = 0.35
ax.bar(x - width/2, valeurs1, width, label='Série 1')
ax.bar(x + width/2, valeurs2, width, label='Série 2')
ax.set_xticks(x)
ax.set_xticklabels(catégories)
ax.legend()
```

### 3. Nuage de points (Scatter plot)

```python
fig, ax = plt.subplots()

x = np.random.randn(100)
y = np.random.randn(100)
couleurs = np.random.randn(100)

scatter = ax.scatter(x, y, c=couleurs, cmap='viridis', 
                     s=100, alpha=0.6, edgecolors='black')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Nuage de points')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Valeur')

plt.show()
```

**Paramètres** :
- `c` : couleur ou tableau de valeurs
- `s` : taille des points
- `alpha` : transparence
- `cmap` : colormap
- `edgecolors` : couleur des bordures

### 4. Histogramme (Histogram)

```python
fig, ax = plt.subplots()

données = np.random.normal(loc=100, scale=15, size=1000)

ax.hist(données, bins=30, color='green', alpha=0.7, edgecolor='black')

ax.set_xlabel('Valeurs')
ax.set_ylabel('Fréquence')
ax.set_title('Distribution normale')
ax.grid(axis='y', alpha=0.3)

plt.show()
```

**Paramètres** :
- `bins` : nombre ou position des intervalles
- `density` : normaliser pour obtenir une densité de probabilité

### 5. Diagramme en boîte (Box plot)

```python
fig, ax = plt.subplots()

données = [np.random.normal(0, std, 100) for std in range(1, 4)]

ax.boxplot(données, labels=['A', 'B', 'C'])
ax.set_ylabel('Valeurs')
ax.set_title('Diagramme en boîte')
ax.grid(axis='y', alpha=0.3)

plt.show()
```

### 6. Graphique en camembert (Pie chart)

```python
fig, ax = plt.subplots()

valeurs = [30, 25, 20, 25]
labels = ['A', 'B', 'C', 'D']
couleurs = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

ax.pie(valeurs, labels=labels, colors=couleurs, autopct='%1.1f%%')
ax.set_title('Répartition')

plt.show()
```

---

## Personnalisation Avancée

### 1. Styles et thèmes

```python
# Voir les styles disponibles
print(plt.style.available)

# Utiliser un style
plt.style.use('seaborn-v0_8-darkgrid')

# Ou spécifier directement
fig, ax = plt.subplots(style='ggplot')
```

### 2. Couleurs et colormaps

```python
# Couleurs nommées
couleurs_disponibles = plt.colormaps()

# Utiliser une colormap
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
for i in range(5):
    y = np.sin(x + i)
    ax.plot(x, y, label=f'sin(x + {i})')

ax.legend()
plt.show()
```

### 3. Annotations

```python
fig, ax = plt.subplots()

x = np.linspace(0, 10, 50)
y = x**2

ax.plot(x, y)

# Annoter un point
ax.annotate('Maximum', xy=(10, 100), xytext=(8, 80),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12, color='red')

# Texte simple
ax.text(5, 50, 'Zone d\'intérêt', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.show()
```

### 4. Polices et texte

```python
from matplotlib import font_manager

fig, ax = plt.subplots()

# Titre avec style
ax.set_title('Titre en gras', fontsize=14, fontweight='bold')
ax.set_xlabel('Label X', fontsize=12, style='italic')
ax.set_ylabel('Label Y', fontsize=12)

# Modifier les ticks
ax.tick_params(axis='x', labelsize=10, rotation=45)

plt.tight_layout()
plt.show()
```

### 5. Légende avancée

```python
fig, ax = plt.subplots()

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')
ax.plot(x, np.tan(x), label='tan(x)')

# Légende personnalisée
ax.legend(loc='upper left', frameon=True, shadow=True, 
         ncol=2, fontsize=10, title='Fonctions')

plt.show()
```

---

## Gestion des Figures et Sous-graphiques

### 1. Plusieurs sous-graphiques (subplots)

```python
# Grille 2x2
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Accéder aux axes
axes[0, 0].plot([1, 2, 3], [1, 4, 2])
axes[0, 0].set_title('Graphique 1')

axes[0, 1].scatter([1, 2, 3], [1, 4, 2])
axes[0, 1].set_title('Graphique 2')

axes[1, 0].bar(['A', 'B', 'C'], [1, 4, 2])
axes[1, 0].set_title('Graphique 3')

axes[1, 1].hist(np.random.randn(100))
axes[1, 1].set_title('Graphique 4')

plt.tight_layout()
plt.show()
```

### 2. Mise en page flexible (GridSpec)

```python
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, :])  # Premier ligne complète
ax2 = fig.add_subplot(gs[1, :-1])  # Deuxième ligne sauf dernière colonne
ax3 = fig.add_subplot(gs[1:, -1])  # Dernier colonne pour 2 lignes
ax4 = fig.add_subplot(gs[-1, 0])
ax5 = fig.add_subplot(gs[-1, 1])

# Remplir avec des données...
```

### 3. Espacement et dimensioning

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Dimensioning
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1,
                    wspace=0.3, hspace=0.4)

# Ou utiliser tight_layout
plt.tight_layout()
```

---

## Graphiques Spécialisés

### 1. Heatmap

```python
import numpy as np

data = np.random.rand(5, 5)

fig, ax = plt.subplots()
im = ax.imshow(data, cmap='coolwarm')

ax.set_xticks(np.arange(5))
ax.set_yticks(np.arange(5))
ax.set_xticklabels(['A', 'B', 'C', 'D', 'E'])
ax.set_yticklabels(['1', '2', '3', '4', '5'])

# Ajouter les valeurs
for i in range(5):
    for j in range(5):
        text = ax.text(j, i, f'{data[i, j]:.2f}',
                      ha="center", va="center", color="black")

plt.colorbar(im, ax=ax)
plt.show()
```

### 2. Graphique en violon (Violin plot)

```python
fig, ax = plt.subplots()

données = [np.random.normal(0, std, 100) for std in range(1, 4)]
ax.violinplot(données, positions=[1, 2, 3], showmeans=True)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['A', 'B', 'C'])
ax.set_ylabel('Valeurs')
ax.set_title('Violin plot')
ax.grid(axis='y', alpha=0.3)

plt.show()
```

### 3. Contour plot

```python
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2)

fig, ax = plt.subplots()
contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
ax.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5)

plt.colorbar(contour, ax=ax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Contour plot')

plt.show()
```

### 4. Graphique en nuage 3D

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Données
theta = np.linspace(0, 4*np.pi, 100)
z = np.linspace(0, 10, 100)
x = 5 * np.cos(theta)
y = 5 * np.sin(theta)

ax.plot(x, y, z, color='blue')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Courbe 3D')

plt.show()
```

---

## Bonnes Pratiques

### 1. Structure recommandée

```python
import matplotlib.pyplot as plt
import numpy as np

def créer_graphique(données_x, données_y):
    """Crée et retourne un graphique."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tracer
    ax.plot(données_x, données_y, color='blue', linewidth=2)
    
    # Personnaliser
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Mon Graphique', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig, ax

# Utilisation
x = np.linspace(0, 10, 100)
y = np.sin(x)
fig, ax = créer_graphique(x, y)
plt.show()
```

### 2. Sauvegarde de haute qualité

```python
fig, ax = plt.subplots()
ax.plot(x, y)

# Sauvegarder en haute résolution
plt.savefig('graphique.png', dpi=300, bbox_inches='tight')
plt.savefig('graphique.pdf', bbox_inches='tight')
plt.savefig('graphique.svg', bbox_inches='tight')
```

### 3. Gestion des ressources

```python
# Fermer les figures pour libérer la mémoire
plt.close(fig)

# Ou fermer tout
plt.close('all')

# Contexte manager
with plt.ioff():  # Interactive mode off
    fig, ax = plt.subplots()
    ax.plot(x, y)
```

### 4. Travailler avec les axes

```python
fig, ax = plt.subplots()

ax.plot(x, y)

# Limites des axes
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)

# Échelle logarithmique
ax.set_yscale('log')
ax.set_xscale('log')

# Aspect ratio
ax.set_aspect('equal')

# Inverser un axe
ax.invert_yaxis()
```

---

## Intégration avec NumPy et Pandas

### 1. Avec NumPy

```python
import matplotlib.pyplot as plt
import numpy as np

# Données NumPy
x = np.linspace(0, 10, 100)
y = np.sin(x)
erreurs = np.random.normal(0, 0.1, len(x))

fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=erreurs, fmt='o', capsize=5, alpha=0.6)
ax.plot(x, y, 'b-', linewidth=2)

plt.show()
```

### 2. Avec Pandas

```python
import matplotlib.pyplot as plt
import pandas as pd

# DataFrame
df = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=30),
    'Ventes': np.random.randint(100, 500, 30),
    'Coûts': np.random.randint(50, 300, 30)
})

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df['Date'], df['Ventes'], label='Ventes', marker='o')
ax.plot(df['Date'], df['Coûts'], label='Coûts', marker='s')

ax.set_xlabel('Date')
ax.set_ylabel('Montant (€)')
ax.set_title('Ventes vs Coûts')
ax.legend()
ax.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 3. Graphique depuis un DataFrame

```python
# Directement avec pandas
df.plot(x='Date', y=['Ventes', 'Coûts'], figsize=(12, 6))

# Ou avec plus de contrôle
fig, ax = plt.subplots()
df.plot(x='Date', y='Ventes', kind='bar', ax=ax)
```

---

## Exemples Complets

### Exemple 1 : Dashboard de statistiques

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Créer les données
ventes = np.random.normal(1000, 200, 365)
mois = np.arange(12)
ventes_mensuelles = np.random.randint(20000, 40000, 12)

# Créer la figure
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Série temporelle (2 colonnes)
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(ventes, color='blue', linewidth=1.5)
ax1.fill_between(range(len(ventes)), ventes, alpha=0.3)
ax1.set_title('Ventes quotidiennes', fontweight='bold')
ax1.set_ylabel('Montant (€)')
ax1.grid(True, alpha=0.3)

# 2. Statistique (1 colonne)
ax2 = fig.add_subplot(gs[0, 2])
stats = [np.mean(ventes), np.median(ventes), np.std(ventes)]
ax2.bar(['Moyenne', 'Médiane', 'Écart-type'], stats, color=['red', 'green', 'blue'])
ax2.set_title('Statistiques', fontweight='bold')
ax2.tick_params(axis='x', rotation=45)

# 3. Ventes mensuelles (2 colonnes)
ax3 = fig.add_subplot(gs[1, :2])
ax3.bar(mois + 1, ventes_mensuelles, color='skyblue', edgecolor='navy')
ax3.set_title('Ventes mensuelles', fontweight='bold')
ax3.set_xlabel('Mois')
ax3.set_ylabel('Montant (€)')
ax3.set_xticks(mois + 1)

# 4. Distribution (1 colonne)
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(ventes, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax4.set_title('Distribution', fontweight='bold')
ax4.set_ylabel('Fréquence')

# 5. Corrélation (3 colonnes)
ax5 = fig.add_subplot(gs[2, :])
x = np.random.randn(100)
y = 2*x + np.random.randn(100)
ax5.scatter(x, y, alpha=0.6, s=50)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
ax5.plot(x, p(x), "r--", linewidth=2, label='Tendance')
ax5.set_title('Corrélation X vs Y', fontweight='bold')
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.legend()
ax5.grid(True, alpha=0.3)

plt.suptitle('Dashboard Commercial 2024', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Exemple 2 : Analyse comparative

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Données
catégories = ['Produit A', 'Produit B', 'Produit C', 'Produit D']
2024 = [100, 150, 120, 200]
2023 = [80, 140, 100, 180]

# Graphique 1 : Barres groupées
ax = axes[0, 0]
x = np.arange(len(catégories))
width = 0.35
ax.bar(x - width/2, 2023, width, label='2023', color='lightblue')
ax.bar(x + width/2, 2024, width, label='2024', color='darkblue')
ax.set_ylabel('Ventes')
ax.set_title('Comparaison 2023 vs 2024')
ax.set_xticks(x)
ax.set_xticklabels(catégories, rotation=15)
ax.legend()

# Graphique 2 : Variation en %
ax = axes[0, 1]
variation = [(2024[i] - 2023[i]) / 2023[i] * 100 for i in range(len(catégories))]
colors = ['green' if v > 0 else 'red' for v in variation]
ax.bar(catégories, variation, color=colors, alpha=0.7)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Variation (%)')
ax.set_title('Variation annuelle')
ax.tick_params(axis='x', rotation=15)

# Graphique 3 : Tendance
ax = axes[1, 0]
trimestres = ['Q1', 'Q2', 'Q3', 'Q4']
ventes_2024_trim = [100, 120, 140, 200]
ax.plot(trimestres, ventes_2024_trim, marker='o', linewidth=2, markersize=8)
ax.fill_between(range(len(trimestres)), ventes_2024_trim, alpha=0.3)
ax.set_title('Tendance 2024')
ax.set_ylabel('Ventes')
ax.grid(True, alpha=0.3)

# Graphique 4 : Répartition
ax = axes[1, 1]
ax.pie(2024, labels=catégories, autopct='%1.1f%%', startangle=90)
ax.set_title('Part de marché 2024')

plt.tight_layout()
plt.savefig('analyse_comparative.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Ressources utiles

- **Documentation officielle** : https://matplotlib.org/
- **Galerie** : https://matplotlib.org/stable/gallery/index.html
- **NumPy** : https://numpy.org/
- **Pandas** : https://pandas.pydata.org/
- **Colormaps** : https://matplotlib.org/stable/tutorials/colors/colormaps.html

---

## Résumé des commandes essentielles

| Fonction | Utilité |
|----------|---------|
| `plt.figure()` | Créer une nouvelle figure |
| `plt.subplots()` | Créer figure + axes |
| `ax.plot()` | Graphique en courbe |
| `ax.bar()` / `ax.barh()` | Barres verticales/horizontales |
| `ax.scatter()` | Nuage de points |
| `ax.hist()` | Histogramme |
| `ax.imshow()` | Heatmap |
| `ax.set_title()` | Titre |
| `ax.set_xlabel()` / `ax.set_ylabel()` | Labels des axes |
| `ax.legend()` | Légende |
| `ax.grid()` | Grille |
| `plt.savefig()` | Sauvegarder |
| `plt.show()` | Afficher |
| `plt.tight_layout()` | Ajuster l'espacement |

---

**Bon apprentissage avec Matplotlib ! 📊**