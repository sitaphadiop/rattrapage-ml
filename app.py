import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

def format_fcfa(value):
    """Format a numeric value as FCFA with space thousands separators and no decimals."""
    try:
        n = int(round(float(value)))
    except Exception:
        return f"{value} FCFA"
    s = f"{n:,}".replace(",", " ")
    return f"{s} FCFA"

st.set_page_config(page_title="Projet Machine Learning", layout="wide")

st.title("Projet de Rattrapage - Machine Learning")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sélectionnez la tâche", ["Régression (Prix de vente)", "Classification (État du véhicule)"])

#regression
if page == "Régression (Prix de vente)":
    st.header("Prédiction du Prix de Vente (Selling Price)")
    st.write("Ce modèle prédit le prix de vente d'un véhicule basé sur ses caractéristiques.")
    
    # Formulaire de saisie pour la Régression
    kms_driven = st.number_input("Kilométrage (Kms_Driven)", min_value=0, value=50000)
    present_price = st.number_input("Prix actuel en concession (Present_Price)", min_value=0.0, value=10.0)
    fuel_type = st.selectbox("Type de carburant (Fuel_Type)", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Type de vendeur (Seller_Type)", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    age = st.number_input("Âge du véhicule (Age)", min_value=0, max_value=30, value=5)
    unit_choice = st.selectbox("Unité de sortie du modèle", ["Auto", "Unités (FCFA)", "kFCFA (x1 000)", "MFCFA (x1 000 000)"], index=0)
    
    if st.button("Prédire le prix"):
        # Vérifier que les fichiers existent avant de charger
        model_path = "models/model_regression.joblib"
        scaler_path = "models/scaler_reg.joblib"
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.error("Fichiers du modèle de régression introuvables. Exécutez create_dummy_models.py pour les générer.")
            st.write("Vous pouvez lancer : `python create_dummy_models.py` dans le dossier du projet.")
        else:
            # Chargement du modèle et du scaler de régression
            model_reg = joblib.load(model_path)
            scaler_reg = joblib.load(scaler_path)

            # Encodage manuel des variables (selon l'ordre de vos LabelEncoders)
            fuel_enc = {"CNG": 0, "Diesel": 1, "Petrol": 2}[fuel_type]
            seller_enc = {"Dealer": 0, "Individual": 1}[seller_type]
            trans_enc = {"Automatic": 0, "Manual": 1}[transmission]

            features = np.array([[kms_driven, present_price, fuel_enc, seller_enc, trans_enc, age]])
            features_scaled = scaler_reg.transform(features)

            prediction = model_reg.predict(features_scaled)
            # Determine multiplier: Auto applies heuristic scaling when outputs look like small unit numbers
            multiplier = 1
            applied_note = None
            if unit_choice == "kFCFA (x1 000)":
                multiplier = 1_000
            elif unit_choice == "MFCFA (x1 000 000)":
                multiplier = 1_000_000
            elif unit_choice == "Auto":
                # Heuristic: if model predicts a small number (<100), interpret as thousands (kFCFA)
                raw = float(prediction[0])
                if abs(raw) < 100:
                    multiplier = 1_000
                    applied_note = "(Auto-scaled: x1 000)"

            converted = float(prediction[0]) * multiplier
            price_fcfa = format_fcfa(converted)
            st.success(f"Le prix de vente estimé est de : {price_fcfa}")

#classification
else:
    st.header("Prédiction de l'État de la Voiture (Classification)")
    st.write("Ce modèle évalue l'état d'un véhicule à partir de ses données Expat Car.")
    
   
    # Exemple générique à adapter selon votre dataset :
    km = st.number_input("Kilométrage", min_value=0, value=30000)
    annee = st.number_input("Année", min_value=2000, max_value=2026, value=2018)
    
    if st.button("Évaluer l'état"):
        clf_path = "models/model_classif.joblib"
        scaler_clf_path = "models/scaler_classif.joblib"
        if not os.path.exists(clf_path) or not os.path.exists(scaler_clf_path):
            st.error("Fichiers du modèle de classification introuvables. Exécutez create_dummy_models.py pour les générer.")
            st.write("Vous pouvez lancer : `python create_dummy_models.py` dans le dossier du projet.")
        else:
            model_clf = joblib.load(clf_path)
            scaler_clf = joblib.load(scaler_clf_path)

            features = np.array([[km, annee]]) # Adapter selon les variables réelles
            features_scaled = scaler_clf.transform(features)

            prediction = model_clf.predict(features_scaled)
            st.success(f"Résultat de la classification (État) : {prediction[0]}")
