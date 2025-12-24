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
airlines = [
    "Air India", "IndiGo", "Vistara", "SpiceJet", "GO_FIRST"
]

sources = [
    "Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"
]

destinations = [
    "Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"
]

classes = ["Economy", "Business"]

stops = ["zero", "one", "two_or_more"]

departure_times = [
    "Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"
]

arrival_times = [
    "Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"
]
# Dictionnaire des durées moyennes de vol entre villes
duration_dict = {
    ("Delhi", "Mumbai"): 120,
    ("Delhi", "Bangalore"): 150,
    ("Delhi", "Kolkata"): 130,
    ("Delhi", "Hyderabad"): 140,
    ("Delhi", "Chennai"): 160,
    ("Mumbai", "Bangalore"): 100,
    ("Mumbai", "Kolkata"): 140,
    ("Mumbai", "Hyderabad"): 110,
    ("Mumbai", "Chennai"): 120,
    ("Bangalore", "Kolkata"): 160,
    ("Bangalore", "Hyderabad"): 90,
    ("Bangalore", "Chennai"): 80,
    ("Kolkata", "Hyderabad"): 150,
    ("Kolkata", "Chennai"): 140,
    ("Hyderabad", "Chennai"): 70
}

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
