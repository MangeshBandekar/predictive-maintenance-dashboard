import numpy as np
import pandas as pd

# Create a random-number generator
random_generator = np.random.default_rng(42)

# Create 500 pretend sensor readings
number_of_readings = 500

# Create the date and time for every reading
timestamps = pd.date_range(
    start="2026-09-01 08:00:00",
    periods=number_of_readings,
    freq="10min"
)

# Pretend the tractor begins with 120 runtime hours
runtime_hours = 120 + np.arange(number_of_readings) * (10 / 60)

# Create normal temperature readings
temperature = random_generator.normal(
    loc=75,
    scale=4,
    size=number_of_readings
)

# Create normal vibration readings
vibration = random_generator.normal(
    loc=2.5,
    scale=0.4,
    size=number_of_readings
)

# At first, mark every reading as normal
condition = np.full(
    number_of_readings,
    "NORMAL",
    dtype=object
)

# Select 20 readings to represent serious equipment problems
maintenance_positions = random_generator.choice(
    number_of_readings,
    size=20,
    replace=False
)

# Find the remaining positions
remaining_positions = np.setdiff1d(
    np.arange(number_of_readings),
    maintenance_positions
)

# Select 35 readings to represent smaller warnings
warning_positions = random_generator.choice(
    remaining_positions,
    size=35,
    replace=False
)

# Make warning readings hotter and shakier
temperature[warning_positions] += 10
vibration[warning_positions] += 1.2
condition[warning_positions] = "CHECK SOON"

# Make serious readings much hotter and shakier
temperature[maintenance_positions] += 23
vibration[maintenance_positions] += 3.0
condition[maintenance_positions] = "MAINTENANCE NEEDED"

# Create the data table
tractor_data = pd.DataFrame(
    {
        "timestamp": timestamps,
        "equipment_id": "Tractor_01",
        "runtime_hours": runtime_hours.round(2),
        "temperature_c": temperature.round(2),
        "vibration_mm_s": vibration.round(2),
        "simulated_condition": condition,
    }
)

# Save the table as a CSV spreadsheet
tractor_data.to_csv(
    "tractor_sensor_data.csv",
    index=False
)

print("Success! Pretend tractor data was created.")
print("Number of readings:", len(tractor_data))
print()
print(tractor_data.head())