from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np

# Ce script crée des modèles/scalers "dummy" pour permettre de lancer
# l'application Streamlit si vous n'avez pas encore les vrais fichiers .joblib.

os.makedirs('models', exist_ok=True)

# Régression : app.py attend 6 features
X_reg = np.array([
    [10000, 5.0, 2, 0, 1, 3],
    [50000,10.0, 1, 1, 1, 5],
    [20000, 8.0, 2, 0, 1, 2]
])
y_reg = np.array([8.0, 5.0, 7.0])

scaler_reg = StandardScaler().fit(X_reg)
model_reg = DummyRegressor(strategy='mean').fit(scaler_reg.transform(X_reg), y_reg)

joblib.dump(model_reg, 'models/model_regression.joblib')
joblib.dump(scaler_reg, 'models/scaler_reg.joblib')

# Classification : app.py attend 2 features [km, annee]
X_clf = np.array([
    [30000,2018],
    [50000,2015],
    [15000,2020]
])
y_clf = np.array([1, 0, 1])  # labels d'exemple

scaler_clf = StandardScaler().fit(X_clf)
model_clf = DummyClassifier(strategy='most_frequent').fit(scaler_clf.transform(X_clf), y_clf)

joblib.dump(model_clf, 'models/model_classif.joblib')
joblib.dump(scaler_clf, 'models/scaler_classif.joblib')

print("Fichiers dummy créés dans le dossier models/:")
print(" - models/model_regression.joblib")
print(" - models/scaler_reg.joblib")
print(" - models/model_classif.joblib")
print(" - models/scaler_classif.joblib")
