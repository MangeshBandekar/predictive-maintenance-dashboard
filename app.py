import joblib
import pandas as pd
import streamlit as st

# Set up the webpage
st.set_page_config(
    page_title="Equipment Maintenance Predictor",
    page_icon="🚜",
    layout="centered"
)

# Load the trained machine-learning model
saved_information = joblib.load("maintenance_model.joblib")
model = saved_information["model"]
input_columns = saved_information["input_columns"]

# Page title and explanation
st.title("🚜 Equipment Maintenance Predictor")

st.write(
    "Enter the tractor's current sensor readings. "
    "The machine-learning model will estimate its condition."
)

# Create input boxes
runtime_hours = st.number_input(
    "Runtime Hours",
    min_value=0.0,
    value=150.0,
    step=1.0
)

temperature = st.number_input(
    "Engine Temperature (°C)",
    min_value=0.0,
    value=75.0,
    step=1.0
)

vibration = st.number_input(
    "Vibration (mm/s)",
    min_value=0.0,
    value=2.5,
    step=0.1
)

# Make a prediction when the button is clicked
if st.button("Predict Equipment Condition"):

    current_reading = pd.DataFrame(
        [[runtime_hours, temperature, vibration]],
        columns=input_columns
    )

    prediction = model.predict(current_reading)[0]

    probabilities = model.predict_proba(current_reading)[0]
    confidence = max(probabilities) * 100

    st.subheader("Prediction Result")

    if prediction == "NORMAL":
        st.success("NORMAL — Equipment appears to be operating normally.")

    elif prediction == "CHECK SOON":
        st.warning("CHECK SOON — Schedule an equipment inspection.")

    else:
        st.error(
            "MAINTENANCE NEEDED — Stop and inspect the equipment."
        )

    st.write("Model confidence:", round(confidence, 2), "%")

    st.subheader("Current Sensor Readings")

    st.write("Runtime hours:", runtime_hours)
    st.write("Temperature:", temperature, "°C")
    st.write("Vibration:", vibration, "mm/s")

st.divider()

st.caption(
    "Prototype created using simulated tractor sensor data. "
    "Real-world decisions require inspection by qualified personnel."
)

# Explain how the model makes decisions
st.subheader("How the Model Makes a Decision")

st.write(
    "The model examines runtime hours, engine temperature, "
    "and vibration. It learned their importance from the "
    "simulated training data."
)

importance_table = pd.DataFrame(
    {
        "Measurement": [
            "Runtime Hours",
            "Engine Temperature",
            "Vibration"
        ],
        "Importance": model.feature_importances_
    }
)

importance_table = importance_table.set_index("Measurement")

st.bar_chart(importance_table)

st.write(
    "A larger bar means that measurement had a stronger "
    "influence on the model."
)

# Open the historical tractor data
historical_data = pd.read_csv("tractor_sensor_data.csv")
historical_data["timestamp"] = pd.to_datetime(
    historical_data["timestamp"]
)

st.divider()
st.header("Historical Tractor Sensor Data")

st.subheader("Engine Temperature History")

temperature_history = historical_data.set_index(
    "timestamp"
)[["temperature_c"]]

st.line_chart(temperature_history)

st.subheader("Vibration History")

vibration_history = historical_data.set_index(
    "timestamp"
)[["vibration_mm_s"]]

st.line_chart(vibration_history)

st.subheader("Equipment Condition Summary")

condition_summary = historical_data[
    "simulated_condition"
].value_counts()

st.bar_chart(condition_summary)

with st.expander("View Recent Sensor Readings"):
    st.dataframe(
        historical_data.tail(20),
        width="stretch"
    )