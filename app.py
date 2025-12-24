from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Charger le modèle et les colonnes
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Charger dataset pour les menus déroulants
df_raw = pd.read_csv("Clean_Dataset.csv")
duration_dict = df_raw.groupby(['source_city', 'destination_city'])['duration'].mean().to_dict()
duration_dict = {f"{k[0]}-{k[1]}": v for k, v in duration_dict.items()}

airlines = sorted(df_raw['airline'].unique())
sources = sorted(df_raw['source_city'].unique())
destinations = sorted(df_raw['destination_city'].unique())
classes = sorted(df_raw['class'].unique())
stops = sorted(df_raw['stops'].unique())
departure_times = sorted(df_raw['departure_time'].unique())
arrival_times = sorted(df_raw['arrival_time'].unique())

# Colonnes catégorielles pour get_dummies
cat_columns = ['airline', 'source_city', 'departure_time', 'stops', 
               'arrival_time', 'destination_city', 'class']

# Taux de conversion inr → TND
taux_inr_tnd = 0.096 

@app.route("/", methods=["GET", "POST"])
def index():
    price = ""
    if request.method == "POST":
        # Récupérer les valeurs du formulaire
        input_data = {}
        for col in cat_columns:
            input_data[col] = request.form[col]

        # Récupérer durée et jours avant départ
        input_data['duration'] = float(request.form['duration'])
        input_data['days_left'] = int(request.form['days_left'])
        
        # Créer un DataFrame pour le modèle
        df_input = pd.DataFrame([input_data])
        df_input = pd.get_dummies(df_input)

        # Ajouter les colonnes manquantes pour correspondre au modèle
        for col in model_columns:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[model_columns]

        # Prédire le prix en XOF
        price_inr = model.predict(df_input)[0]

        # Convertir en dinar tunisien et arrondir
        price = round(price_inr * taux_inr_tnd, 2)

    # Renvoyer le template avec toutes les options
    return render_template(
        "index.html",
        price=price,
        airlines=airlines,
        sources=sources,
        destinations=destinations,
        classes=classes,
        stops=stops,
        departure_times=departure_times,
        arrival_times=arrival_times,
        duration_dict=duration_dict
    )

if __name__ == "__main__":
    app.run()
