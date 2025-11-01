import streamlit as st

# --- Callback: clamp date_debut si inférieure à la date_limite intraday ---
def _clamp_date_debut_if_needed():
    lim = st.session_state.get("_date_limite_intraday", None)
    if lim is None:
        return
    # Si l'utilisateur choisit une date antérieure à la limite, on remonte à la limite et on relance
    if st.session_state.get("date_debut") and st.session_state["date_debut"] < lim:
        st.session_state["date_debut"] = lim
        
        st.rerun()

import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration et Données ---
st.set_page_config(layout="wide", initial_sidebar_state="auto")

INTERVALLLES = {
    "Journalier (1d)": "1d",
    "Horaire (1h)": "1h",
    "30 minutes (30m)": "30m",
    "15 minutes (15m)": "15m",
}

# --- Fonctions de Traitement et de Données Utilitaires ---
@st.cache_data
def convert_df_to_csv(df):
    """Convertit le DataFrame en CSV pour le téléchargement."""
    return df.to_csv(sep=';', decimal=',', encoding='utf-8')

@st.cache_data
def charger_donnees(ticker, date_debut, date_fin, intervalle):
    """
    Télécharge les données boursières avec une granularité spécifiée et gère les erreurs de colonnes.
    """
    try:
        # Intraday : utiliser 'period' pour fiabilité
        if intervalle in ["15m", "30m"]:
            data = yf.download(
                ticker,
                period="60d", #limite période yfinance pour infraday 15min/30min
                interval=intervalle,
                progress=False
            )
            # --- Filtrage intraday sur la plage utilisateur (respecte date_debut/date_fin) ---
            try:
                data = data.loc[(data.index.date >= date_debut) & (data.index.date <= date_fin)]
            except Exception:
                pass
        elif intervalle == "1h":
            data = yf.download(
                ticker,
                period="730d", #limite période yfinance pour infraday 1h
                interval=intervalle,
                progress=False
            )
            # --- Filtrage intraday sur la plage utilisateur (respecte date_debut/date_fin) ---
            try:
                data = data.loc[(data.index.date >= date_debut) & (data.index.date <= date_fin)]
            except Exception:
                pass
        else:
            # Journalier : fin inclusive (+1 jour)
            data = yf.download(
                ticker,
                start=date_debut.strftime('%Y-%m-%d'),
                end=(date_fin + datetime.timedelta(days=1)).strftime('%Y-%m-%d'),
                interval=intervalle,
                progress=False
            )

        if data is None or data.empty:
            st.error(f"Aucune donnée reçue pour {ticker} (Période: {date_debut} à {date_fin}, Intervalle: {intervalle}).")
            return pd.DataFrame()
        
        # Aplatir l'éventuel MultiIndex de colonnes
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        # S'assurer que 'Adj Close' existe.
        if 'Adj Close' not in data.columns and 'Close' in data.columns:
            data['Adj Close'] = data['Close']

        # Retirer timezone de l'index si nécessaire
        try:
            data.index = data.index.tz_localize(None)
        except Exception:
            pass
        return data
    except Exception as e:
        st.error(f"Erreur lors du téléchargement pour {ticker}: {e}")
        return pd.DataFrame()
    
        # Flatten éventuellement l'index de colonnes multi-niveaux
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        # S'assurer d'avoir 'Adj Close'
        if 'Adj Close' not in data.columns and 'Close' in data.columns:
            data['Adj Close'] = data['Close']
        # Retirer timezone de l'index si nécessaire
        try:
            data.index = data.index.tz_localize(None)
        except Exception:
            pass
        return data
    except Exception as e:
        st.error(f"Erreur de téléchargement pour {ticker}: {e}")
        return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        if 'Adj Close' not in data.columns and 'Close' in data.columns:
            data['Adj Close'] = data['Close']
        return data
    except Exception as e:
        st.error(f"Erreur de téléchargement pour {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data
def calculer_indicateurs(df, periode_mm_bb, periode_rsi, type_mm, bb_std):
    """Calcule les indicateurs techniques (MM, BB, RSI) sur le DataFrame."""
    data = df.copy() # Évite le SettingWithCopyWarning

    # 1. Validation de la période
    if periode_mm_bb <= 1 or periode_mm_bb > len(data):
        periode_mm_bb = max(2, min(len(data), 20)) # Défaut sécurisé
        st.warning(f"Période MM/BB ajustée à {periode_mm_bb} (hors limites).")

    # 2. Moyenne Mobile (SMA ou EMA)
    if type_mm == "SMA":
        data['MM'] = data['Close'].rolling(window=periode_mm_bb).mean()
    else:
        data['MM'] = data['Close'].ewm(span=periode_mm_bb, adjust=False).mean()

    # 3. Bandes de Bollinger (BB)
    data['Ecart_Type'] = data['Close'].rolling(window=periode_mm_bb).std()
    data['Bande_Sup'] = data['MM'] + (data['Ecart_Type'] * bb_std)
    data['Bande_Inf'] = data['MM'] - (data['Ecart_Type'] * bb_std)

    # 4. Indice de Force Relative (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periode_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periode_rsi).mean()

    # Éviter la division par zéro   
    with np.errstate(divide='ignore', invalid='ignore'):
        RS = gain / loss
        data['RSI'] = 100 - (100 / (1 + RS))
        # Gérer le cas où loss est 0 (RS = infini) -> RSI = 100
        data['RSI'] = data['RSI'].replace(np.inf, 100)

    return data

# Calcul des métriques
def calculer_metriques(df, is_daily_data, taux_sans_risque):
    """Calcule les métriques de performance et de risque."""
    data = df.copy()
    metriques = {
        'rendement_total': 0.0, 
        'max_drawdown': 0.0, 
        'volatilite': 0.0, 
        'sharpe_ratio': np.nan, 
        'perf_annualisee': 0.0
    }

    if len(data) < 2:
        return metriques

    # Rendement Total
    premier_prix = data["Adj Close"].iloc[0]
    dernier_prix = data["Adj Close"].iloc[-1]
    metriques['rendement_total'] = (dernier_prix / premier_prix - 1) * 100

    # Performance annualisée (CAGR) basée sur la durée réelle entre la 1ère et la dernière observation
    try:
        debut = data.index[0]
        fin = data.index[-1]
        annees = (fin - debut).days / 365.25
        if annees > 0 and data["Adj Close"].iloc[0] > 0:
            cagr = (data["Adj Close"].iloc[-1] / data["Adj Close"].iloc[0]) ** (1/annees) - 1
            metriques['perf_annualisee'] = cagr * 100
        else:
            metriques['perf_annualisee'] = 0.0
    except Exception:
        metriques['perf_annualisee'] = 0.0

    # Drawdown Maximal (MaxDD)
    data['Peak'] = data["Adj Close"].cummax()
    data['Drawdown'] = (data["Adj Close"] / data['Peak']) - 1
    metriques['max_drawdown'] = data['Drawdown'].min() * 100

    # Rendement pour Volatilité et Sharpe
    data["Rendement"] = data["Close"].pct_change()

    # Volatilité et Ratio de Sharpe (Seulement si données journalières)    
    if is_daily_data and not data["Rendement"].dropna().empty:
        rendements_non_na = data["Rendement"].dropna()
        volatilite_journaliere = rendements_non_na.std()

        metriques['volatilite'] = volatilite_journaliere * (252**0.5) * 100

        rendement_moyen_quotidien = rendements_non_na.mean()
        taux_sans_risque_journalier = (1 + taux_sans_risque)**(1/252) - 1

        if volatilite_journaliere != 0:
            rendement_excedentaire = rendement_moyen_quotidien - taux_sans_risque_journalier
            metriques['sharpe_ratio'] = (rendement_excedentaire / volatilite_journaliere) * np.sqrt(252)

    return metriques

# --- Interface Utilisateur (Sidebar pour les Inputs) ---
st.title("📈 Analyse Multi-Actifs")
st.markdown("""
Cette application permet d'analyser l'évolution d'un titre financier avec des graphiques interactifs et des métriques avancées.
""")

st.sidebar.title("Paramètres d'Analyse")
ticker_principal = st.sidebar.text_input("Ticker Principal :", "AAPL").upper()

# Saisie des Tickers
st.sidebar.subheader("1. Choix des Tickers")
ticker_principal = st.sidebar.text_input(
    "Ticker Principal (ex: AAPL, NVDA, AIR.PA) :",
    value="AAPL"
).upper()

# Récupération du nom complet de l’entreprise
try:
    ticker_obj = yf.Ticker(ticker_principal)
    company_name = ticker_obj.info.get("longName", ticker_principal)
except Exception:
    company_name = ticker_principal

# --------------------------------------------------------------------
#  Remplacement de st.text_input par st.multiselect
# --------------------------------------------------------------------
tickers_comparaison = st.sidebar.multiselect(
    "Tickers de Comparaison (Optionnel) :",
    options=["MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "AIR.PA", "BTC-USD"], # Exemples pré-remplis
    default=["MSFT"]
)
# --------------------------------------------------------------------

# Récupération des noms complets des entreprises de comparaison
noms_comparaison = {}
for t in tickers_comparaison:
    try:
        t_obj = yf.Ticker(t)
        noms_comparaison[t] = t_obj.info.get("longName", t)
    except Exception:
        noms_comparaison[t] = t

# --- Initialisation des dates en session_state (ajout minimal) ---
_today_tmp = datetime.date.today()
if "date_debut" not in st.session_state:
    st.session_state.date_debut = _today_tmp - datetime.timedelta(days=730)
if "date_fin" not in st.session_state:
    st.session_state.date_fin = _today_tmp

# Choix des Périodes Spécifiques
st.sidebar.subheader("2. Période et Granularité")
date_debut = st.sidebar.date_input("Date de début :", key="date_debut")
date_fin = st.sidebar.date_input("Date de fin :", key="date_fin")

# Choix de l'intervalle
choix_intervalle_label = st.sidebar.selectbox("Granularité des données :", list(INTERVALLLES.keys()), index=0)
# Emplacement réservé pour le warning (sous la granularité)
warn_slot = st.sidebar.empty()

intervalle_yf = INTERVALLLES[choix_intervalle_label]
is_daily_data = (intervalle_yf == "1d")
intervalle_yf = INTERVALLLES[choix_intervalle_label]
is_daily_data = (intervalle_yf == "1d")

# Logique de validation des dates basée sur l'intervalle
today = datetime.date.today()
date_limite = today # Par défaut (pour 1d, pas de limite réelle)

if intervalle_yf in ["15m", "30m"]:
    # Limite de yfinance pour 15m/30m est de 60 jours
    date_limite = today - datetime.timedelta(days=60)
    label_limite = "60 jours"


elif intervalle_yf == "1h":
    # Limite de yfinance pour 1h est de 730 jours
    date_limite = today - datetime.timedelta(days=730)
    label_limite = "730 jours"
else: # "1d"
    # Pour '1d', on garde une limite large (pas de limite yf réelle)
    date_limite = today - datetime.timedelta(days=50*365) 
    label_limite = "Maximum (Journalier)"


# (nettoyé) clamp local supprimé — on garde un seul warning top

# (nettoyé) bloc de warning doublon supprimé — on conserve le warning top
date_fin = st.session_state.date_fin

# S'assurer que la date de fin n'est pas antérieure à la date de début (MAJ auto + refresh)
if st.session_state.date_fin < st.session_state.date_debut:
    st.session_state.date_fin = today
    st.rerun()
# Valeurs locales à partir de l'état (pour la suite du code)
date_debut = st.session_state.date_debut
date_fin = st.session_state.date_fin

# Paramètres des Indicateurs Techniques
st.sidebar.subheader("3. Indicateurs Techniques")

# Choix SMA / EMA
type_mm = st.sidebar.selectbox("Type de Moyenne Mobile :", ["SMA", "EMA"], index = 0)

# Période MM / BB
mm_label = "Période MM / BB"
if intervalle_yf == "1d": mm_label += " (jours)"
elif intervalle_yf == "1h": mm_label += " (heures)"
else: mm_label += " (périodes)"
periode_mm_bb = st.sidebar.slider(mm_label, 10, 100, 20)

# Écart-type BB
bb_std = st.sidebar.slider("Écarts-types Bandes de Bollinger :", 1.0, 4.0, 2.0, 1.0)

# Période RSI
periode_rsi = st.sidebar.slider("Période RSI :", 7, 30, 14)

# --- Seuils RSI personnalisables ---
try:
    st.sidebar.markdown("##### RSI - Seuil Surachat/Survente")
    seuil_surachat = st.sidebar.slider("Seuil de surachat (RSI)", 51, 99, 70, 1, key="rsi_overbought")
    seuil_survente = st.sidebar.slider("Seuil de survente (RSI)", 1, 49, 30, 1, key="rsi_oversold")
except Exception:
    # Valeurs par défaut si sidebar ou sliders non disponibles à cet endroit
    seuil_surachat = 70
    seuil_survente = 30

# Paramètres de Risque
st.sidebar.subheader("4. Paramètres de Risque")
taux_sans_risque = st.sidebar.number_input("Taux sans risque annualisé (%) :", value=2.0, min_value=0.0, max_value=10.0) / 100

# Contrôles d'affichage
st.sidebar.markdown("---")
st.sidebar.subheader("5. Affichage & Exportation")

# --- Section: Métriques ---
st.sidebar.markdown("##### Affichage des Métriques")
show_performance = st.sidebar.checkbox("Afficher les Métriques de Risque", value=True)

# --- Section: Graphique Principal ---
st.sidebar.markdown("##### Options du Graphique Principal")
chart_type = st.sidebar.radio(
    "Type de Graphique", 
    ["Ligne (Close)", "Candlestick (OHLC)"], 
    index=1 # Défaut sur Candlestick
)
show_ma = st.sidebar.checkbox(f"Afficher Moyenne Mobile ({type_mm})", value=True)
show_bb = st.sidebar.checkbox("Afficher les Bandes de Bollinger", value=True)

# --- Section: Sous-Graphiques ---
st.sidebar.markdown("##### Options des Sous-Graphiques")
show_rsi_subplot = st.sidebar.checkbox("Afficher RSI", value=True)

# --- Section: Export ---
st.sidebar.markdown("##### Exportation")
download_placeholder = st.sidebar.empty()

# --- Logique de Traitement Principale ---

# Chargement des données du ticker principal
data_p = charger_donnees(ticker_principal, date_debut, date_fin, intervalle_yf)

# --- Warning avec la date EFFECTIVE de début (intraday & daily) ---
try:
    if data_p is not None and not data_p.empty:
        effective_start = data_p.index.min().date()

        # Limites YF en fonction de l'intervalle
        _today_top = datetime.date.today()
        if intervalle_yf in ["15m", "30m"]:
            limit_date = _today_top - datetime.timedelta(days=60)
            _label_top = "60 jours"
        elif intervalle_yf == "1h":
            limit_date = _today_top - datetime.timedelta(days=730)
            _label_top = "730 jours"
        else:
            limit_date = _today_top - datetime.timedelta(days=50*365)  # très large pour daily
            _label_top = "Maximum (Journalier)"

        if intervalle_yf in ["1h", "30m", "15m"]:
            # (1) Cas "clamp" intraday : date_debut antérieure à la limite YF
            if date_debut < limit_date:
                # Afficher la date réellement utilisée (effective_start) pour cohérence graphe/table
                warn_slot.warning(
                    f"Pour l'intervalle '{choix_intervalle_label}', l'historique est limité à {_label_top}. "
                    f"La date de début a été ajustée au {effective_start.strftime('%Y-%m-%d')}."
                )
            # (2) Pas de donnée le jour choisi (week-end/jour férié)
            elif effective_start > date_debut:
                warn_slot.warning(
                    f"Aucune donnée intrajournalière le {date_debut.strftime('%Y-%m-%d')} (week‑end ou jour férié). "
                    f"Les données commencent le {effective_start.strftime('%Y-%m-%d')}."
                )
        else:
            # Daily : seulement le cas "pas de donnée ce jour" (week-end/jour férié / marché fermé)
            if effective_start > date_debut:
                warn_slot.warning(
                    f"Aucune donnée le {date_debut.strftime('%Y-%m-%d')} (week‑end ou jour férié). "
                    f"Les données commencent le {effective_start.strftime('%Y-%m-%d')}."
                )
except Exception:
    pass


# Vérification critique
if data_p.empty or 'Close' not in data_p.columns:
    st.error(f"Impossible de charger les données pour {ticker_principal}. Vérifiez le ticker et la plage de dates.")
    st.stop()

# Calcul des indicateurs et métriques
data_p = calculer_indicateurs(data_p, periode_mm_bb, periode_rsi, type_mm, bb_std)
metriques_p = calculer_metriques(data_p, is_daily_data, taux_sans_risque)

# --------------------------------------------------------------------
# Logique de chargement pour plusieurs tickers de comparaison
# --------------------------------------------------------------------

# Normalisation du ticker principal (toujours nécessaire pour l'onglet 2)
if not data_p.empty:
    data_p['Rendement_Norm'] = (data_p['Adj Close'] / data_p['Adj Close'].iloc[0]) * 100

# Dictionnaire pour stocker les dataframes de comparaison
data_comparaison_dict = {}

if tickers_comparaison: # Si la liste n'est pas vide
    for ticker in tickers_comparaison:
        # Mettre les tickers en majuscule (au cas où l'utilisateur ajoute manuellement)
        ticker = ticker.upper() 

        # Éviter de re-charger le ticker principal
        if ticker == ticker_principal:
            st.warning(f"Le ticker principal {ticker} est déjà affiché.")
            continue

        data_c = charger_donnees(ticker, date_debut, date_fin, intervalle_yf)

        if not data_c.empty:
            # Normaliser les rendements pour la comparaison
            data_c['Rendement_Norm'] = (data_c['Adj Close'] / data_c['Adj Close'].iloc[0]) * 100
            data_comparaison_dict[ticker] = data_c # Ajouter au dictionnaire
        else:
            st.warning(f"Impossible de charger les données pour le ticker de comparaison {ticker}.")
# --------------------------------------------------------------------


# --- Bouton de Téléchargement ---
csv_data = convert_df_to_csv(data_p)
filename = f"{ticker_principal}_{intervalle_yf}_{date_debut}_a_{date_fin}.csv"

with download_placeholder:
    st.download_button(
        label="⬇️ Télécharger (CSV)",
        data=csv_data,
        file_name=filename,
        mime='text/csv',
    )

# --- Affichage du Dashboard (AVEC ONGLETS) ---

st.header(f"Analyse pour {company_name}")

tab1, tab2, tab3 = st.tabs(["📊 Analyse Principale", "🆚 Comparaison", "📋 Données Brutes"])

with tab1:
    # -------------------------------------------------------------
    # Affichage Conditionnel 1 : Métriques de Performance et Risque
    # -------------------------------------------------------------
    if show_performance:
        st.subheader("Métriques de Performance et Risque")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Peformance Cumulée Totale (%)", f"{metriques_p['rendement_total']:.2f}%")
        with col2:
            st.metric("Performance annualisée (%)", f"{metriques_p['perf_annualisee']:.2f}%")
        with col3:
            if is_daily_data:
                st.metric("Volatilité Annualisée (%)", f"{metriques_p['volatilite']:.2f}%")
            else : 
                st.metric("Volatilité Annualisée (%)", "N/A (Infradaily)")
        with col4:
            st.metric("Max Drawdown (%)", f"{metriques_p['max_drawdown']:.2f}%")
        with col5:
            if not np.isnan(metriques_p['sharpe_ratio']):
                st.metric("Ratio de Sharpe", f"{metriques_p['sharpe_ratio']:.2f}")
            else:
                st.metric("Ratio de Sharpe", "N/A (Infradaily)")
        st.markdown("---")

    # -------------------------------------------------------------
    # Graphique Principal Interactif (Plotly)
    # -------------------------------------------------------------

    st.subheader("📉 Analyse Graphique Interactive")

    rows = 2 if show_rsi_subplot else 1
    row_heights = [0.7, 0.3] if show_rsi_subplot else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=[f"Cours de {company_name}"] + (["RSI"] if show_rsi_subplot else [])
    )

    data_plot = data_p.dropna(subset=["Open", "High", "Low", "Close"])

    # --- Graphique principal ---

    # --- Candlestick ---
    if chart_type == "Candlestick (OHLC)" and not data_plot.empty:
        fig.add_trace(go.Candlestick(
            x=data_plot.index, open=data_plot["Open"], high=data_plot["High"],
            low=data_plot["Low"], close=data_plot["Close"],
            name="OHLC", increasing_line_color="green", decreasing_line_color="red"
        ), row=1, col=1)
    # --- Close Price ---
    elif chart_type == "Ligne (Close)":
        fig.add_trace(go.Scatter(
            x=data_plot.index, y=data_plot["Close"], mode="lines",
            line=dict(color="#1f77b4", width=2), name="Cours (Close)"
        ), row=1, col=1)

    # --- Moyenne Mobile ---
    if show_ma and "MM" in data_plot:
        fig.add_trace(go.Scatter(
            x=data_plot.index, y=data_plot["MM"], mode="lines",
            line=dict(color="orange", width=1.5), name=f"{type_mm} {periode_mm_bb}"
        ), row=1, col=1)
    
    # --- Bolliger ---
    if show_bb and {"Bande_Sup", "Bande_Inf"}.issubset(data_plot.columns):
        # Bande Sup
        fig.add_trace(go.Scatter(
            x=data_plot.index, y=data_plot["Bande_Sup"], mode="lines",
            line=dict(color="red", width=1, dash="dot"), name=f"Bande Sup. (+{bb_std}σ)"
        ), row=1, col=1)
        # Bande Inf
        fig.add_trace(go.Scatter(
            x=data_plot.index, y=data_plot["Bande_Inf"], mode="lines",
            line=dict(color="green", width=1, dash="dot"), name=f"Bande Inf. (-{bb_std}σ)"
        ), row=1, col=1)

    # --- RSI ---
    if show_rsi_subplot and "RSI" in data_plot:
        fig.add_trace(go.Scatter(
            x=data_plot.index, y=data_plot["RSI"], mode="lines",
            line=dict(color="purple", width=1.5), name=f"RSI {periode_rsi}"
        ), row=2, col=1)
        # Seuils Surchat/Survente
        fig.add_trace(go.Scatter(
            x=data_plot.index, y=[seuil_surachat]*len(data_plot),
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            name=f"Seuil Surachat ({seuil_surachat})"
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=data_plot.index, y=[seuil_survente]*len(data_plot),
            mode='lines',
            line=dict(color='green', width=1, dash='dash'),
            name=f"Seuil Survente ({seuil_survente})"
        ), row=2, col=1)

        fig.update_yaxes(range=[0, 100], row=2, col=1, title_text="RSI", fixedrange=True)

    # 🔒 Range slider totalement désactivé
    fig.update_layout(xaxis_rangeslider_visible=False)

    fig.update_layout(
        height=600,
        title=f"Analyse de {company_name} ({choix_intervalle_label})",
        hovermode="x unified",
        legend_title_text="Indicateurs",
        xaxis_title="Date",
        yaxis_title="Prix ($)"
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------
# Logique d'affichage pour plusieurs tickers
# --------------------------------------------------------------------
with tab2:
    st.subheader("Comparaison des Rendements Cumulées (Base 100)")

    # Vérifier si la liste de comparaison (de la sidebar) n'est pas vide
    if tickers_comparaison:
        fig_comp = go.Figure()

        # 1. Ajouter le ticker principal (toujours)
        fig_comp.add_trace(go.Scatter(
            x=data_p.index, 
            y=data_p['Rendement_Norm'], 
            mode='lines', 
            name=ticker_principal, 
            line=dict(width=3) # Ligne principale plus épaisse
        ))

        # 2. Boucler sur les tickers de comparaison qui ont été chargés
        for ticker, data_c in data_comparaison_dict.items():
            nom_c = noms_comparaison.get(ticker, ticker)
            fig_comp.add_trace(go.Scatter(
                x=data_c.index, 
                y=data_c['Rendement_Norm'], 
                mode='lines', 
                name=nom_c, 
                line=dict(width=1.5, dash='dot') # Lignes de comparaison
            ))

        # Créer un titre dynamique
        compaison_title = f"Comparaison : {company_name} vs {', '.join([noms_comparaison.get(t, t) for t in tickers_comparaison])}"

        fig_comp.update_layout(
            title=compaison_title,
            xaxis_title="Date", 
            yaxis_title="Rendement Normalisé (Base 100)",
            hovermode="x unified", 
            legend_title_text="Tickers"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    else:
        st.info("Sélectionnez un ou plusieurs 'Tickers de Comparaison' dans la barre latérale pour activer ce graphique.")
# --------------------------------------------------------------------

with tab3:
    st.subheader(f"📋 Données Historiques et Indicateurs pour {company_name}")

    # Filtrer les colonnes pour n'afficher que les pertinentes
    cols_a_afficher = [
        "Open", "High", "Low", "Close", "Adj Close", "Volume",
        "MM", "Bande_Sup", "Bande_Inf", "RSI", "Rendement"
    ]
    cols_disponibles = [c for c in cols_a_afficher if c in data_p.columns]

    # Remplacement de st.dataframe par st.data_editor
    st.data_editor(
        data_p[cols_disponibles].sort_index(ascending=False), 
        use_container_width=True, 
        num_rows="fixed" # ou "dynamic" si vous préférez
    )
