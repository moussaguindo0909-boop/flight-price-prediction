import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pickle

# Charger les données
df = pd.read_csv("Clean_Dataset.csv")

# Supprimer les colonnes inutiles
df = df.drop(columns=['Unnamed: 0', 'flight'])

# Transformer toutes les colonnes texte en nombres
# Liste des colonnes catégorielles
cat_columns = ['airline', 'source_city', 'departure_time', 'stops', 
               'arrival_time', 'destination_city', 'class']

df = pd.get_dummies(df, columns=cat_columns, drop_first=True)

# Séparer X et y
X = df.drop("price", axis=1)
y = df["price"]

# Séparer en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Modèle Random Forest
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MAE :", mean_absolute_error(y_test, y_pred))
print("R2 :", r2_score(y_test, y_pred))

# Sauvegarder le modèle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Sauvegarder les colonnes X pour aligner les nouvelles données
with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns, f)