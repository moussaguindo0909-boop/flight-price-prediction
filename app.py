from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Charger modèle et colonnes
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Données pour formulaires
airlines = ["Air India", "IndiGo", "Vistara", "SpiceJet", "GO_FIRST"]
sources = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
destinations = sources
classes = ["Economy", "Business"]
stops = ["zero", "one", "two_or_more"]

departure_times = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
arrival_times = departure_times

# Durées moyennes
duration_dict = {
    "Delhi-Mumbai": 2.0,
    "Delhi-Bangalore": 2.5,
    "Delhi-Kolkata": 2.2,
    "Delhi-Hyderabad": 2.3,
    "Delhi-Chennai": 2.7,
    "Mumbai-Bangalore": 1.7,
    "Mumbai-Kolkata": 2.3,
    "Mumbai-Hyderabad": 1.8,
    "Mumbai-Chennai": 2.0,
    "Bangalore-Kolkata": 2.7,
    "Bangalore-Hyderabad": 1.5,
    "Bangalore-Chennai": 1.3,
    "Kolkata-Hyderabad": 2.5,
    "Kolkata-Chennai": 2.3,
    "Hyderabad-Chennai": 1.2
}

cat_columns = [
    "airline", "source_city", "departure_time",
    "stops", "arrival_time", "destination_city", "class"
]

# INR → TND
TAUX_INR_TND = 0.096

@app.route("/", methods=["GET", "POST"])
def index():
    price = None

    if request.method == "POST":
        try:
            input_data = {col: request.form[col] for col in cat_columns}

            duration = float(request.form["duration"])
            days_left = int(request.form["days_left"])

            # Sécurité
            if duration <= 0 or duration > 20:
                duration = 3

            input_data["duration"] = duration
            input_data["days_left"] = days_left

            df = pd.DataFrame([input_data])
            df = pd.get_dummies(df)

            for col in model_columns:
                if col not in df.columns:
                    df[col] = 0

            df = df[model_columns]

            price_inr = model.predict(df)[0]
            price = round(price_inr * TAUX_INR_TND, 2)

        except Exception as e:
            print("Erreur prédiction :", e)
            price = None

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
    app.run(debug=True)
