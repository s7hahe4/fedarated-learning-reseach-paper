import pandas as pd

# Define the filename
file_path = 'diabetes.csv'

try:
    # Load the dataset from the CSV file
    df = pd.read_csv(file_path)

    print("--- 1. Successfully loaded diabetes.csv ---")
    print("\n")

    # --- Data Exploration ---
    print("--- 2. First 5 Rows of the Dataset ---")
    print(df.head())
    print("\n")

    print("--- 3. Dataset Information (Columns and Data Types) ---")
    df.info()
    print("\n")

    print("--- 4. Statistical Summary of the Dataset ---")
    print(df.describe())
    print("\n")

    print("--- 5. Distribution of the 'Outcome' Variable ---")
    print(df['Outcome'].value_counts())
    print("\n")
    
    print("--- Step 1 Complete ---")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure 'diabetes.csv' is in the same folder as your Python script.")