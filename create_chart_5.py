import matplotlib.pyplot as plt
import numpy as np

# --- Our Collected Results ---
model_names = [
    'Centralized', 
    'Basic FL', 
    'DP FL (Low Noise)', 
    'DP FL (Medium Noise)', 
    'DP FL (High Noise)'
]
accuracies = [77.92, 72.08, 70.13, 63.64, 72.08]

# --- Creating the Bar Chart ---
plt.figure(figsize=(10, 6)) # Set the figure size
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
bars = plt.bar(model_names, accuracies, color=colors)

# --- Adding Labels and Title ---
plt.ylabel('Test Accuracy (%)')
plt.title('Model Accuracy vs. Privacy Level')
plt.ylim(0, 100) # Set y-axis to go from 0 to 100

# Add the accuracy value on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

# --- Save the Chart ---
file_name = 'results_chart.png'
plt.savefig(file_name)

print(f"--- Chart saved successfully as '{file_name}'! ---")
print("You can find the image in your D:\\python folder.")