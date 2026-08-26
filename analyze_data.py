import pandas as pd
import matplotlib.pyplot as plt

# Open the pretend sensor data
data = pd.read_csv("tractor_sensor_data.csv")

# Change the timestamp column into real date and time values
data["timestamp"] = pd.to_datetime(data["timestamp"])

# Separate the warning and maintenance readings
warnings = data[data["simulated_condition"] == "CHECK SOON"]
maintenance = data[
    data["simulated_condition"] == "MAINTENANCE NEEDED"
]

# Count each equipment condition
print("Equipment condition summary:")
print(data["simulated_condition"].value_counts())

# Create an area for two graphs
figure, graphs = plt.subplots(
    2,
    1,
    figsize=(12, 8),
    sharex=True
)

# Temperature graph
graphs[0].plot(
    data["timestamp"],
    data["temperature_c"],
    color="orange",
    label="Temperature"
)

graphs[0].scatter(
    warnings["timestamp"],
    warnings["temperature_c"],
    color="gold",
    label="Check Soon"
)

graphs[0].scatter(
    maintenance["timestamp"],
    maintenance["temperature_c"],
    color="red",
    label="Maintenance Needed"
)

graphs[0].set_title("Tractor Engine Temperature")
graphs[0].set_ylabel("Temperature (C)")
graphs[0].legend()
graphs[0].grid(True)

# Vibration graph
graphs[1].plot(
    data["timestamp"],
    data["vibration_mm_s"],
    color="blue",
    label="Vibration"
)

graphs[1].scatter(
    warnings["timestamp"],
    warnings["vibration_mm_s"],
    color="gold",
    label="Check Soon"
)

graphs[1].scatter(
    maintenance["timestamp"],
    maintenance["vibration_mm_s"],
    color="red",
    label="Maintenance Needed"
)

graphs[1].set_title("Tractor Vibration")
graphs[1].set_xlabel("Date and Time")
graphs[1].set_ylabel("Vibration (mm/s)")
graphs[1].legend()
graphs[1].grid(True)

# Make the graph neat
plt.tight_layout()

# Save the graph as an image
plt.savefig(
    "tractor_sensor_graph.png",
    dpi=300
)

# Display the graph
plt.show()