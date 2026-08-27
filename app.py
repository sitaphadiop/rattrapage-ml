import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Projet Machine Learning", layout="wide")

st.title("Projet de Rattrapage - Machine Learning")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sélectionnez la tâche", ["Régression (Prix de vente)", "Classification (État du véhicule)"])

# -----------------------------------------------------------------------------
# PAGE 1 : RÉGRESSION
# -----------------------------------------------------------------------------
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
    
    if st.button("Prédire le prix"):
        # Chargement du modèle et du scaler de régression
        model_reg = joblib.load("models/model_regression.joblib")
        scaler_reg = joblib.load("models/scaler_reg.joblib")
        
        # Encodage manuel des variables (selon l'ordre de vos LabelEncoders)
        fuel_enc = {"CNG": 0, "Diesel": 1, "Petrol": 2}[fuel_type]
        seller_enc = {"Dealer": 0, "Individual": 1}[seller_type]
        trans_enc = {"Automatic": 0, "Manual": 1}[transmission]
        
        features = np.array([[kms_driven, present_price, fuel_enc, seller_enc, trans_enc, age]])
        features_scaled = scaler_reg.transform(features)
        
        prediction = model_reg.predict(features_scaled)
        st.success(f"Le prix de vente estimé est de : {prediction[0]:.2f} (en unités de valeur)")

# -----------------------------------------------------------------------------
# PAGE 2 : CLASSIFICATION
# -----------------------------------------------------------------------------
else:
    st.header("Prédiction de l'État de la Voiture (Classification)")
    st.write("Ce modèle évalue l'état d'un véhicule à partir de ses données Expat Car.")
    
    # Ajoutez ici les champs correspondant aux variables de votre dataset de classification
    # Exemple générique à adapter selon votre dataset :
    km = st.number_input("Kilométrage", min_value=0, value=30000)
    annee = st.number_input("Année", min_value=2000, max_value=2026, value=2018)
    
    if st.button("Évaluer l'état"):
        model_clf = joblib.load("models/model_classif.joblib")
        scaler_clf = joblib.load("models/scaler_classif.joblib")
        
        features = np.array([[km, annee]]) # Adapter selon les variables réelles
        features_scaled = scaler_clf.transform(features)
        
        prediction = model_clf.predict(features_scaled)
        st.success(f"Résultat de la classification (État) : {prediction[0]}")
