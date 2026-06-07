import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Features & labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("Dataset shape:", df.shape)
print("Class distribution:\n", y.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model (slightly stronger than before)
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=700,
    activation='relu'
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
print("Saved updated model.pkl")