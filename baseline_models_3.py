import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

# Suppress verbose TensorFlow logging and other warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration ---
NUM_CLIENTS = 10
file_path = 'diabetes.csv'

# --- 1. Load and Prepare Data (Same as Step 2) ---
df = pd.read_csv(file_path)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Preprocessing: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("--- Data Loaded and Preprocessed ---")

# --- Helper Function to Create a Model ---
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 2. Centralized Model Training ---
print("\n--- Training Centralized Model (Baseline 1) ---")
centralized_model = create_model()
centralized_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
loss, centralized_accuracy = centralized_model.evaluate(X_test, y_test, verbose=0)
print(f"Centralized Model Test Accuracy: {centralized_accuracy * 100:.2f}%")

# --- 3. Federated Model Simulation (Baseline 2) ---
print("\n--- Simulating and Training Basic Federated Model (Baseline 2) ---")

# Simulate federated clients with training data
shuffled_indices = np.random.permutation(len(X_train))
X_train_shuffled = X_train[shuffled_indices]
y_train_shuffled = y_train.iloc[shuffled_indices]

client_data_X = np.array_split(X_train_shuffled, NUM_CLIENTS)
client_data_y = np.array_split(y_train_shuffled, NUM_CLIENTS)

# Initialize a global model on the server
global_model = create_model()
communication_rounds = 10

# Federated Averaging Loop
for round_num in range(communication_rounds):
    client_weights = []
    for i in range(NUM_CLIENTS):
        # Each client trains on its local data
        local_model = create_model()
        local_model.set_weights(global_model.get_weights())
        local_model.fit(client_data_X[i], client_data_y[i], epochs=5, batch_size=16, verbose=0)
        client_weights.append(local_model.get_weights())
    
    # --- FIXED: More robust weight averaging ---
    # Create a new list to store the averaged weights for each layer
    new_weights = []
    # Get the number of layers in the model
    num_layers = len(global_model.get_weights())
    # Loop through each layer
    for layer_index in range(num_layers):
        # Get the weights for the current layer from all clients
        layer_weights = np.array([client[layer_index] for client in client_weights])
        # Average the weights for the current layer
        avg_layer_weights = np.mean(layer_weights, axis=0)
        new_weights.append(avg_layer_weights)

    # Update the global model with the new averaged weights
    global_model.set_weights(new_weights)
    
    # Optional: Print progress
    if (round_num + 1) % 2 == 0:
        print(f"  - Communication Round {round_num + 1}/{communication_rounds} complete...")

# Evaluate the final federated model
loss, federated_accuracy = global_model.evaluate(X_test, y_test, verbose=0)
print(f"Basic Federated Model Test Accuracy: {federated_accuracy * 100:.2f}%")

print("\n--- Step 3 Complete ---")