from flask import Flask, request, render_template
import pickle

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

            # Convert inputs to float for prediction
            features = [float(user_inputs[f]) for f in feature_names]

            prediction = model.predict([features])[0]
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

    # Render the initial page with empty inputs
    return render_template("index.html", feature_names=feature_names, user_inputs={})

if __name__ == "__main__":
    app.run(debug=True)
