# Rattrapage ML - Application Streamlit

Ce dépôt contient une application Streamlit pour deux tâches : prédiction du prix de vente (régression) et classification de l'état d'un véhicule.

Contenu du dépôt :
- app.py : application Streamlit (déjà ajoutée)
- requirements.txt : dépendances Python
- models/ : dossier pour vos fichiers .joblib (modèles et scalers)
- encoders/ : dossier pour les fichiers d'encodage (.joblib)

Instructions pour lancer l'application :
1. Cloner le dépôt :
   git clone https://github.com/sitaphadiop/rattrapage-ml.git
2. Installer les dépendances :
   pip install -r requirements.txt
3. Exécuter Streamlit :
   streamlit run app.py

Important : les boutons de prédiction chargeront les fichiers depuis models/*.joblib et encoders/*.joblib. Tant que ces fichiers ne sont pas présents, l'application lèvera une erreur au moment du chargement. Vous pouvez :
- fournir les fichiers .joblib et je les ajouterai pour vous, ou
- uploader les fichiers directement via l'interface GitHub (Add file → Upload files).
