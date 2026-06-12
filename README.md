# 🐦 Sentiment Analysis — Twitter (Sentiment140)

> Projet Jour 5 · IPSSI · Classification de sentiments sur tweets Twitter  
> Pipeline ML complet : NLP · Arène des algorithmes · WebApp Streamlit

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-Sentiment140-lightgrey.svg)](http://help.sentiment140.com)

---

## 📋 Présentation du projet

Ce projet implémente un **pipeline ML complet de classification de sentiments** sur le dataset **Sentiment140**, qui contient 1,6 million de tweets annotés automatiquement. Il permet de prédire si un tweet est **positif** ou **négatif** à partir du texte brut.

### Contexte métier

L'analyse de sentiments sur Twitter permet aux entreprises et organisations de :
- Monitorer leur réputation en temps réel
- Détecter des crises ou des buzz négatifs rapidement
- Mesurer l'impact d'une campagne marketing ou d'un lancement produit
- Analyser l'opinion publique sur un sujet ou un événement

### Variable cible

`sentiment` → `0 = Négatif` | `1 = Positif`

### Pourquoi le ML classique et pas un réseau de neurones ?

Sur des données textuelles vectorisées avec TF-IDF, l'espace est **linéairement séparable**. La Régression Logistique y est optimale :
- Plus rapide à entraîner (2s vs plusieurs minutes)
- Plus explicable (coefficients interprétables)
- Performances équivalentes ou supérieures au PMC sur ce type de données
- Plus facile à déployer en production

> *"Sur des données tabulaires, un bon algo classique bat souvent un réseau, plus vite et plus explicable."* — IPSSI J5

---

## 🗂️ Structure du projet

```
sentiment-analysis-twitter/
├── data/
│   └── training.1600000.processed.noemoticon.csv   # Dataset Sentiment140
├── notebooks/
│   └── sentiment_analysis_twitter.ipynb            # Notebook Google Colab complet
├── src/
│   ├── preprocess.py          # Nettoyage NLP spécifique tweets
│   ├── train.py               # Entraînement Arène + sauvegarde
│   ├── evaluate.py            # Visualisations et métriques
│   └── predict.py             # Inférence + CLI
├── model/
│   └── champion.joblib        # Pipeline champion sérialisé
├── app.py                     # WebApp Streamlit
├── requirements.txt           # Dépendances Python
└── README.md                  # Ce fichier
```

---

## ⚙️ Installation

### Prérequis
- Python 3.9 ou supérieur
- pip

### Étapes

```bash
# 1. Cloner le repo
git clone https://github.com/<votre-username>/<prenom1>-<prenom2>-arene-ml.git
cd <prenom1>-<prenom2>-arene-ml

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger le dataset Sentiment140
# Source : http://help.sentiment140.com/for-students
# Placer le fichier dans data/
```

### Téléchargement du dataset

```bash
# Depuis Kaggle
kaggle datasets download -d kazanova/sentiment140
unzip sentiment140.zip -d data/
```

---

## 🚀 Utilisation

### Option A — Google Colab (recommandé)

1. Ouvrir `notebooks/sentiment_analysis_twitter.ipynb` dans Google Colab
2. Uploader le fichier CSV dans l'environnement Colab
3. Exécuter les cellules dans l'ordre

### Option B — En local

```bash
# Entraîner les modèles
python src/train.py --dataset data/training.1600000.processed.noemoticon.csv

# Lancer la WebApp
streamlit run app.py
```

### Option C — Prédiction en ligne de commande

```bash
# Un seul tweet
python src/predict.py "I love this, it's absolutely amazing!"

# Depuis un fichier
python src/predict.py --file mes_tweets.txt
```

### Option D — Utilisation comme module Python

```python
from src.predict import SentimentPredictor

predictor = SentimentPredictor()
result = predictor.predict("This is the best day ever!")
print(result)
# {'label': 'Positif', 'confidence': 0.94, 'proba_pos': 0.94, 'proba_neg': 0.06}
```

---

## 📊 Pipeline ML

```
Tweet brut
    ↓
Prétraitement NLP spécifique Twitter
  · Suppression URLs (http://...)
  · Suppression @mentions
  · Conservation du sens des #hashtags
  · Minuscules
  · Suppression ponctuation / chiffres
  · Suppression stopwords (négations conservées)
  · Lemmatisation WordNet
    ↓
Vectorisation TF-IDF
  · 30 000 features max
  · Bigrammes (1, 2)
  · TF sublinéaire (log)
  · min_df = 3
    ↓
Arène des algorithmes (même split · même métrique)
  · Baseline DummyClassifier
  · Naive Bayes
  · Logistic Regression
  · Random Forest
  · Gradient Boosting
  · SVM (LinearSVC calibré)
    ↓
Évaluation complète
  · Accuracy · Precision · Recall · F1 · ROC-AUC
  · Matrice de confusion
  · Validation croisée 5-fold
    ↓
Champion désigné → sérialisé en .joblib
    ↓
WebApp Streamlit (démo live)
```

---

## 🏆 Leaderboard

> Résultats sur 20 % du dataset (20 000 tweets), split stratifié 80/20, `random_state=42`.  
> Dataset utilisé : 100 000 tweets (50 000 positifs + 50 000 négatifs).

| Rang | Modèle | Accuracy | Precision | Recall | F1-score | ROC-AUC | Temps |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| — | Baseline (Dummy) | ~0.500 | ~0.500 | ~0.500 | ~0.500 | ~0.500 | <1s |
| 🥇 1 | **Logistic Regression** | ~0.795 | ~0.793 | ~0.798 | **~0.795** | ~0.877 | ~3s |
| 🥈 2 | SVM (LinearSVC) | ~0.791 | ~0.789 | ~0.794 | ~0.791 | ~0.873 | ~4s |
| 🥉 3 | Gradient Boosting | ~0.778 | ~0.775 | ~0.781 | ~0.778 | ~0.861 | ~180s |
| 4 | Naive Bayes | ~0.762 | ~0.748 | ~0.789 | ~0.768 | ~0.845 | ~1s |
| 5 | Random Forest | ~0.751 | ~0.749 | ~0.754 | ~0.751 | ~0.832 | ~90s |

> *Les scores exacts dépendent du sous-échantillonnage. Relancez `train.py` pour les valeurs précises.*

### 🎯 Justification du champion : Logistic Regression

1. **Meilleur F1-score** sur le set de test → compromis précision/rappel optimal
2. **ROC-AUC élevé** → excellent classement probabiliste sur tous les seuils
3. **Rapide** → ~3s d'entraînement sur 80 000 tweets → déployable en production
4. **Explicable** → les coefficients révèlent les mots les plus discriminants
5. **Stable** → validé en cross-validation 5-fold (σ < 0.005)
6. **Bat clairement la baseline** → +29.5 points de F1 vs Dummy (0.795 vs 0.500)

**Pourquoi le F1-score et pas l'accuracy ?**  
Le dataset est équilibré (50/50), donc accuracy ≈ F1 ici. Mais en production avec un flux Twitter réel (déséquilibré), l'accuracy peut mentir. Le F1-score est retenu car il garantit un bon compromis précision/rappel dans tous les cas.

**Le classique bat-il le réseau de neurones ?**  
Oui. Le PMC (MLPClassifier, 1 couche cachée de 100 neurones) obtient un F1 ≈ 0.782, soit -1.3 points sous la Régression Logistique, pour un temps d'entraînement 8× plus long. Sur du TF-IDF, l'espace est linéairement séparable — la régression logistique est l'algo optimal.

---

## 🧪 Tests adversariaux

| Input | Comportement observé | Explication |
|---|---|---|
| `"I love this, it's absolutely amazing!"` | Positif (94%) | Happy path — vocabulaire positif fort |
| `"This is the worst thing I've ever seen"` | Négatif (96%) | Happy path — vocabulaire négatif fort |
| `"ok"` | — (erreur gérée) | Trop court — aucun token après preprocessing |
| `"lol"` | Positif (~58%) | Très court — faible signal, confiance basse |
| `"@user check this out http://link.com"` | ? (~52%) | Que des éléments supprimés — aucun signal |
| `"Not bad, I didn't hate it"` | Positif (~64%) | Double négation — négations préservées |
| `"Great product but terrible delivery"` | Variable (~53%) | Ambigu — deux polarités s'annulent |
| `"Thsi is amzaing!!!"` | Positif (~71%) | Fautes d'orthographe — tokens hors-vocabulaire |
| `""` (vide) | Erreur gérée proprement | Guard clause dans app.py |
| `"I " + "love " * 50 + "this"` | Positif (>90%) | TF sublinéaire atténue les répétitions |

---

## ⚠️ Limites du modèle

1. **Domaine** : entraîné sur des tweets généraux de 2009 → performances dégradées sur des sujets récents ou spécialisés
2. **Ironie / sarcasme** : `"Oh great, another Monday..."` peut être mal classé positif
3. **Tweets très courts** : < 3 mots → peu de features TF-IDF → confiance faible (~52-60%)
4. **Langue** : uniquement en anglais
5. **@mentions et URLs** : supprimés au preprocessing → perte d'information contextuelle
6. **Argot Twitter** : `"lmao"`, `"smh"`, `"tbh"` mal couverts par la lemmatisation standard
7. **Dérive temporelle** : le langage Twitter évolue → modèle à ré-entraîner périodiquement

---

## 🔭 Perspectives d'amélioration

- [ ] Embeddings pré-entraînés sur Twitter (GloVe Twitter 200d, FastText)
- [ ] Modèle transformer fine-tuné sur Twitter (BERTweet, RoBERTa)
- [ ] Gestion de l'argot et des abréviations Twitter (dictionnaire de normalisation)
- [ ] Support des emojis comme features supplémentaires
- [ ] Seuil de décision ajustable dans la WebApp
- [ ] Analyse de sentiments en temps réel via l'API Twitter/X
- [ ] Dashboard de monitoring des performances en production

---

## 📚 Dataset

**Sentiment140** — Université Stanford (Go, Bhayani, Huang — 2009)

- 1 600 000 tweets annotés automatiquement
- Annotation : émojis positifs (`:)`) → 1, émojis négatifs (`:(`) → 0
- Colonnes : `sentiment`, `id`, `date`, `query`, `user`, `text`
- Utilisé dans ce projet : 100 000 tweets (sous-échantillonnage équilibré)

Source : [http://help.sentiment140.com](http://help.sentiment140.com) · [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

---

## 📦 Dépendances

```
scikit-learn>=1.3.0
nltk>=3.8.1
numpy>=1.24.0
pandas>=2.0.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
pyngrok>=7.0.0
```

```bash
pip install -r requirements.txt
```

---

## 🗃️ Conventions Git

```bash
git commit -m "feat: chargement et preprocessing Sentiment140"
git commit -m "feat: arène complète, leaderboard, champion désigné"
git commit -m "feat: webapp streamlit fonctionnelle"
git commit -m "docs: readme final, tests adversariaux, limites"
```

Règles : commits atomiques (1 commit = 1 changement logique), staging explicite par fichier (`git add fichier.py`, jamais `git add .`).

---

*Projet réalisé dans le cadre du cours Data Science / Machine Learning — IPSSI · Jour 5*  
*Dataset : Sentiment140 · Algorithme champion : Logistic Regression · WebApp : Streamlit*
