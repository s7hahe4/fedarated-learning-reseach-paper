import pandas as pd
import numpy as np

# --- Configuration ---
NUM_CLIENTS = 10  # Let's simulate 10 different hospitals
file_path = 'diabetes.csv'

# --- 1. Load the Dataset ---
try:
    df = pd.read_csv(file_path)
    print(f"--- Successfully loaded {file_path} ---")
except FileNotFoundError:
    print(f"Error: Make sure '{file_path}' is in the same folder as your script.")
    exit()

# --- 2. Separate Features (X) and Target (y) ---
# X contains all columns except 'Outcome'
X = df.drop('Outcome', axis=1)
# y contains only the 'Outcome' column
y = df['Outcome']

print("\n--- Data separated into Features (X) and Target (y) ---")
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# --- 3. Simulate Federated Clients ---
# This part splits the dataset into smaller pieces for each client

# Shuffle the data to ensure it's randomly distributed
shuffled_indices = np.random.permutation(len(df))
X_shuffled = X.iloc[shuffled_indices]
y_shuffled = y.iloc[shuffled_indices]

# Split the shuffled data among the number of clients
# np.array_split will handle cases where the data doesn't divide evenly
client_data_X = np.array_split(X_shuffled, NUM_CLIENTS)
client_data_y = np.array_split(y_shuffled, NUM_CLIENTS)

# Create a list to hold each client's complete dataset
clients = []
for i in range(NUM_CLIENTS):
    clients.append({'X': client_data_X[i], 'y': client_data_y[i]})

print(f"\n--- Created a simulation with {NUM_CLIENTS} clients ---")

# --- 4. Verify the Simulation ---
# We'll check the first client to see what their data looks like
print(f"Data for Client 0:")
print(f"  - Number of records (rows): {len(clients[0]['X'])}")
print(f"  - Number of features (columns): {clients[0]['X'].shape[1]}")

print("\n--- Step 2 Complete ---")