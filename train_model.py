import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Open the tractor sensor data
data = pd.read_csv("tractor_sensor_data.csv")

# These are the measurements the model will examine
input_columns = [
    "runtime_hours",
    "temperature_c",
    "vibration_mm_s"
]

# X contains the sensor measurements
X = data[input_columns]

# y contains the correct equipment conditions
y = data["simulated_condition"]

# Use 80 percent of the data for learning
# Use 20 percent for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create the machine-learning model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Teach the model using the training data
model.fit(X_train, y_train)

# Ask the model to predict the test data
predictions = model.predict(X_test)

# Check how many predictions were correct
accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print()
print("Model accuracy:", round(accuracy * 100, 2), "%")
print()
print("Detailed results:")
print(classification_report(y_test, predictions))

# Display what measurements were most important
print("Measurement importance:")

for column, importance in zip(
    input_columns,
    model.feature_importances_
):
    print(column, ":", round(importance * 100, 2), "%")

# Save the trained model
joblib.dump(
    {
        "model": model,
        "input_columns": input_columns
    },
    "maintenance_model.joblib"
)

print()
print("The trained model was saved successfully!")