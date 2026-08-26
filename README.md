# Agricultural Equipment Predictive Maintenance Dashboard

## Project Overview

This project is a Python-based predictive maintenance prototype for agricultural equipment.

The system examines three equipment measurements:

- Runtime hours
- Engine temperature
- Vibration

A machine-learning model uses these measurements to classify the equipment condition as:

- NORMAL
- CHECK SOON
- MAINTENANCE NEEDED

It currently uses simulated tractor sensor data to demonstrate how the complete monitoring system will work.

## Live Dashboard

Use the public dashboard here:

https://predictive-maintenance-dashboard-d37brxcebaf2xao2ampbgv.streamlit.app/

No Python installation is required to use the online dashboard.

## How the System Works

1. `generate_data.py` creates simulated tractor sensor readings.
2. The readings are saved in `tractor_sensor_data.csv`.
3. `analyze_data.py` analyzes the readings and creates historical graphs.
4. `train_model.py` trains a Random Forest machine-learning model.
5. The trained model is saved as `maintenance_model.joblib`.
6. `app.py` provides an interactive Streamlit dashboard.
7. A user enters the current equipment readings.
8. The model predicts the equipment condition.

## Dashboard Inputs

The dashboard requires three values:

| Input | Meaning |
|---|---|
| Runtime Hours | Total number of hours the equipment has operated |
| Engine Temperature | Current engine temperature in degrees Celsius |
| Vibration | Current vibration measurement in millimeters per second |

## Example Tests

### Normal Operation

- Runtime: 150 hours
- Temperature: 75°C
- Vibration: 2.5 mm/s

Expected result: `NORMAL`

### Check Soon

- Runtime: 150 hours
- Temperature: 86°C
- Vibration: 3.8 mm/s

Expected result: `CHECK SOON`

### Maintenance Needed

- Runtime: 150 hours
- Temperature: 100°C
- Vibration: 5.5 mm/s

Expected result: `MAINTENANCE NEEDED`

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Runs the interactive dashboard |
| `generate_data.py` | Creates simulated sensor data |
| `analyze_data.py` | Analyzes data and creates graphs |
| `train_model.py` | Trains and evaluates the model |
| `tractor_sensor_data.csv` | Contains simulated sensor readings |
| `maintenance_model.joblib` | Contains the trained model |
| `tractor_sensor_graph.png` | Shows historical temperature and vibration |
| `requirements.txt` | Lists the required Python packages |
| `.gitignore` | Prevents unnecessary local files from being uploaded |

## Run the Project Locally

Install the required packages:

```bash
pip install -r requirements.txt
