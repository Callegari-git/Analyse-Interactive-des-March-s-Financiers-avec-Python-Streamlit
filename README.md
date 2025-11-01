# 📈 Analyse Interactive des Marchés Financiers avec Python & Streamlit

## 🧠 Objectif du Projet
Ce projet a pour but de créer une **application web locale** en Python, permettant :
- d’analyser un titre financier via des **indicateurs techniques**,
- de calculer des **métriques de performance et de risque**,
- de **comparer plusieurs actifs** (actions, crypto, indices),
- et de visualiser les résultats de façon claire et interactive.

L’application a été réalisée avec **Streamlit**, **Plotly**, **yfinance** et **pandas**.

---

## 🚀 Fonctionnalités Principales

✅ **Téléchargement automatique des données boursières** via Yahoo Finance  
✅ **Visualisation interactive** des cours avec *Plotly* (Ligne ou Chandeliers)  
✅ **Indicateurs techniques intégrés** :
- Moyenne mobile simple (SMA) / exponentielle (EMA)
- Bandes de Bollinger (BB)
- Indice de force relative (RSI)
✅ **Métriques de performance et risque** :
- Rendement total et annualisé  
- Volatilité annualisée  
- Max Drawdown  
- Ratio de Sharpe  
✅ **Comparaison multi-actifs** (ex : AAPL vs MSFT vs BTC-USD)  
✅ **Export CSV** des données analysées  

---

## 🧩 Technologies Utilisées
- **Python 3.10+**
- **Streamlit** (interface web)
- **Plotly** (graphes interactifs)
- **pandas / numpy** (analyse des données)
- **yfinance** (données financières)

---

## 💻 Installation & Exécution Locale

### 1️⃣ Cloner le dépôt
```bash
git clone https://github.com/<ton-utilisateur>/<nom-du-repo>.git
cd <nom-du-repo>
```
### 2️⃣ Créer un environnement virtuel
```bash
python -m venv ven
```
### 3️⃣ Activer l’environnement
Sous Windows :
```bash
venv\Scripts\activate
```
Sous macOS / Linux :
```bash
source venv/bin/activate
```
### 4️⃣ Installer les dépendances
```bash
pip install -r requirements.txt
```
5️⃣ Lancer l’application
```bash
streamlit run Projet.py
```
