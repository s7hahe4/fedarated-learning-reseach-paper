import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
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
    return model

# --- 2. Federated Simulation with Differential Privacy ---
print("\n--- Training Federated Models with Differential Privacy ---")

# We will test different noise levels to get different privacy budgets (epsilons)
noise_multipliers = [1.1, 0.8, 0.5]

for noise_multiplier in noise_multipliers:
    print(f"\n--- Starting training for Noise Multiplier: {noise_multiplier} ---")
    
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
            # Each client creates a local model with a DP optimizer
            local_model = create_model()
            local_model.set_weights(global_model.get_weights())
            
            # --- DP HAPPENS HERE ---
            optimizer = DPKerasAdamOptimizer(
                l2_norm_clip=1.0,
                noise_multiplier=noise_multiplier,
                num_microbatches=1) # Using batch size as microbatches
            
            local_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            local_model.fit(client_data_X[i], client_data_y[i], epochs=CLIENT_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            client_weights.append(local_model.get_weights())
        
        # Aggregate weights on the server
        new_weights = []
        num_layers = len(global_model.get_weights())
        for layer_index in range(num_layers):
            layer_weights = np.array([client[layer_index] for client in client_weights])
            avg_layer_weights = np.mean(layer_weights, axis=0)
            new_weights.append(avg_layer_weights)
            
        global_model.set_weights(new_weights)

    # Calculate the privacy budget (epsilon)
    epsilon = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=len(X_train),
        batch_size=BATCH_SIZE,
        noise_multiplier=noise_multiplier,
        epochs=COMMUNICATION_ROUNDS * CLIENT_EPOCHS,
        delta=1e-5
    )[0]
    
    # Evaluate the final DP federated model
    global_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    loss, dp_accuracy = global_model.evaluate(X_test, y_test, verbose=0)
    
    print(f"  - Privacy Budget (Epsilon): {epsilon:.2f}")
    print(f"  - DP Federated Model Test Accuracy: {dp_accuracy * 100:.2f}%")

print("\n--- Step 4 Complete ---")