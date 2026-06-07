import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Split features and labels
X = df.iloc[:, :-1]   # all columns except last
y = df.iloc[:, -1]    # last column (A/B)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")