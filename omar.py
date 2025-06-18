import os
import random
from io import BytesIO
import base64
from PIL import Image
import streamlit as st
from transformers import pipeline

# --- Fonction utilitaire pour convertir le logo en base64 ---
def logo_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- Configuration de la page ---
st.set_page_config(page_title="Analyse Juridique IA", layout="wide")

# --- CSS personnalisé ---
st.markdown("""
    <style>
    html, body, .stApp {
        background: linear-gradient(to bottom, #000000, #4a148c);
        font-family: 'Georgia', serif;
        color: #ffffff;
    }
    h1, h2, h3, h4, p {
        color: #ffffff !important;
        font-weight: bold;
    }
    textarea, input {
        background-color: #444444 !important;
        color: #ffffff !important;
        font-weight: bold !important;
        border: 1px solid #888888 !important;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] {
        background-color: #8d6e63 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
        font-weight: bold !important;
    }
    .stButton > button {
        background-color: #666666 !important;
        color: #ffffff !important;
        font-weight: bold !important;
        font-size: 18px !important;
        padding: 12px 24px !important;
        border-radius: 10px !important;
        border: none !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #555555 !important;
        transform: scale(1.02);
        transition: all 0.2s;
    }
    </style>
""", unsafe_allow_html=True)

# --- Logo centré ---
dir_script = os.path.dirname(os.path.abspath(__file__))
chemin_logo = os.path.join(dir_script, "logotr.png")  # vérifie bien que ce fichier est dans le même dossier
logo = Image.open(chemin_logo)

st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_to_base64(logo)}" width="120">
    </div>
    """,
    unsafe_allow_html=True
)

# --- Titre ---
st.markdown("""
    <h1 style="text-align: center;">MASTER DROIT ET SÉCURITÉ NUMÉRIQUE</h1>
    <h2 style="text-align: center;">⚖️ Analyse Automatique de Décisions Juridiques</h2>
    <p style="text-align: center; font-size: 17px;">
        Analysez le domaine juridique d'un texte judiciaire (pénal, social...).
    </p>
""", unsafe_allow_html=True)

# --- Citation dans la sidebar ---
citations = [
    "⚖️ *« Le droit est la plus puissante des écoles de l'imagination. »* – Michel Foucault",
    "📚 *« La justice sans la force est impuissante. »* – Blaise Pascal",
    "🧠 *« Un bon avocat connaît la loi. Un grand avocat connaît le juge. »* – Auteur inconnu"
]
st.sidebar.markdown(f"#### 🧾 Citation du jour\n> {random.choice(citations)}")

# --- Zones de saisie ---
col1, col2 = st.columns(2)

with col1:
    texte = st.text_area("📄 Texte de la décision", height=300, placeholder="Collez ici un jugement...")
with col2:
    etiquettes = st.text_input("📌 Branches du droit à tester", "Droit pénal, Droit social, Droit administratif, Droit commercial")

# --- Analyse IA ---
if st.button("🚀 Lancer l'analyse") and texte and etiquettes:
    with st.spinner("Analyse juridique en cours..."):
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        labels = [label.strip() for label in etiquettes.split(",")]
        result = classifier(texte, labels)

    st.markdown("## 📊 Résultats de l'analyse")

    top_label = result['labels'][0]
    top_score = result['scores'][0] * 100

    color = "🟢" if top_score > 75 else "🟠" if top_score > 50 else "🔴"
    bar_color = "#81c784" if top_score > 75 else "#ffb74d" if top_score > 50 else "#e57373"

    st.markdown("### 🏆 Branche juridique dominante")
    st.markdown(f"### {color} {top_label} — {top_score:.2f}%")
    st.markdown(f"""
<div style="background-color:{bar_color}; height:20px; border-radius:10px; width:{top_score}%; margin-bottom:20px;"></div>
    """, unsafe_allow_html=True)

    st.success(f"🔍 Conclusion : Ce texte relève majoritairement du {top_label} ({top_score:.2f}%)")

    st.markdown("---")
    st.markdown("#### 📊 Autres scores")

    autres_labels = result['labels'][1:]
    autres_scores = result['scores'][1:]

    for label, score in zip(autres_labels, autres_scores):
        pct = score * 100
        st.markdown(f"**{label}** : {pct:.2f}%")
        st.markdown(f"""
            <div style="background-color:#cccccc; height:10px; border-radius:8px; width:{pct}%; margin-bottom:8px;"></div>
        """, unsafe_allow_html=True)

# --- Pied de page ---
st.markdown("---")
st.caption("Université Hassan 1er – Analyse Juridique IA – © 2025 | ")
