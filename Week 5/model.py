import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load the dataset
data = sns.load_dataset("penguins")

# Drop rows with missing values
data = data.dropna()

# Encode the target variable (species)
label_encoder = LabelEncoder()
data["species"] = label_encoder.fit_transform(data["species"])

# Select features and target
X = data[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
y = data["species"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
os.makedirs("datasets", exist_ok=True)
model_filename = "datasets/penguin_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(model, file)

# Save feature names
feature_names = X.columns.to_list()
features_filename = "datasets/penguin_features.pkl"
with open(features_filename, "wb") as file:
    pickle.dump(feature_names, file)

# Save label encoder
encoder_filename = "datasets/label_encoder.pkl"
with open(encoder_filename, "wb") as file:
    pickle.dump(label_encoder, file)

# Print model accuracy
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")
