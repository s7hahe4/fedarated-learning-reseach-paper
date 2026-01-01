import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress verbose TensorFlow logging and other warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration ---
NUM_CLIENTS = 10
file_path = 'diabetes.csv'
COMMUNICATION_ROUNDS = 10
CLIENT_EPOCHS = 5
BATCH_SIZE = 16

# --- 1. Load and Prepare Data ---
df = pd.read_csv(file_path)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Helper Function to Create a Model ---
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 2. Federated Simulation with Manual Differential Privacy ---
print("\n--- Training Federated Models with Manual Differential Privacy ---")

# We will test different noise levels. More noise = more privacy.
noise_levels = [0.01, 0.05, 0.1]

for noise_level in noise_levels:
    print(f"\n--- Starting training for Noise Level: {noise_level} ---")
    
    # Simulate federated clients
    shuffled_indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[shuffled_indices]
    y_train_shuffled = y_train.iloc[shuffled_indices]

    client_data_X = np.array_split(X_train_shuffled, NUM_CLIENTS)
    client_data_y = np.array_split(y_train_shuffled, NUM_CLIENTS)

    # Initialize a global model
    global_model = create_model()

    # Federated Averaging Loop
    for round_num in range(COMMUNICATION_ROUNDS):
        client_weights = []
        for i in range(NUM_CLIENTS):
            local_model = create_model()
            local_model.set_weights(global_model.get_weights())
            local_model.fit(client_data_X[i], client_data_y[i], epochs=CLIENT_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            
            # --- MANUAL DP HAPPENS HERE ---
            # Get the weights from the locally trained model
            local_weights = local_model.get_weights()
            noisy_weights = []
            # Add random noise to each layer's weights
            for layer_weights in local_weights:
                noise = np.random.normal(0, noise_level, layer_weights.shape)
                noisy_weights.append(layer_weights + noise)
            
            client_weights.append(noisy_weights)
        
        # Aggregate the noisy weights on the server
        new_weights = []
        num_layers = len(global_model.get_weights())
        for layer_index in range(num_layers):
            layer_weights = np.array([client[layer_index] for client in client_weights])
            avg_layer_weights = np.mean(layer_weights, axis=0)
            new_weights.append(avg_layer_weights)
            
        global_model.set_weights(new_weights)

    # Evaluate the final DP federated model
    loss, dp_accuracy = global_model.evaluate(X_test, y_test, verbose=0)
    
    print(f"  - Noise Level Applied: {noise_level}")
    print(f"  - DP Federated Model Test Accuracy: {dp_accuracy * 100:.2f}%")

print("\n--- Step 4 Complete ---")