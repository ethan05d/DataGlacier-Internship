from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model, feature names, and label encoder
with open("datasets/penguin_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("datasets/penguin_features.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open("datasets/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            user_inputs = {feature: request.form.get(feature) for feature in feature_names}
            
            # Convert inputs to a pandas DataFrame
            features = pd.DataFrame([user_inputs], columns=feature_names)
            features = features.astype(float)  # Ensure numeric data

            prediction = model.predict(features)[0]
            species = label_encoder.inverse_transform([prediction])[0]

            return render_template(
                "index.html",
                prediction=f"Predicted Penguin Species: {species}",
                feature_names=feature_names,
                user_inputs=user_inputs
            )
        except ValueError:
            return render_template(
                "index.html",
                error="Please provide valid numeric inputs.",
                feature_names=feature_names,
                user_inputs=request.form
            )

    return render_template("index.html", feature_names=feature_names, user_inputs={})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON input from the request
        data = request.get_json()
        features = [
            float(data.get("bill_length_mm", 0)),
            float(data.get("bill_depth_mm", 0)),
            float(data.get("flipper_length_mm", 0)),
            float(data.get("body_mass_g", 0))
        ]

        # Predict the species
        prediction = model.predict([features])[0]
        species = label_encoder.inverse_transform([prediction])[0]

        return {"species": species}, 200
    except Exception as e:
        return {"error": str(e)}, 400

if __name__ == "__main__":
    app.run(debug=True)
