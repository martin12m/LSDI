# import glob
# import pandas as pd

# # Path to the folder containing the CSV files
# folder_path = "relational_tables_csv"

# # Find all CSV files in the folder
# csv_files = glob.glob(f"{folder_path}/*.csv")

# if not csv_files:
#     raise FileNotFoundError(f"No CSV files found in the folder: {folder_path}")

# # Combine all CSVs into one DataFrame
# dataframes = []
# for file in csv_files:
#     try:
#         df = pd.read_csv(file)
#         if not df.empty:  # Ensure the file is not empty
#             dataframes.append(df)
#             print(f"{file} loaded successfully. Shape: {df.shape}")
#         else:
#             print(f"{file} is empty. Skipping.")
#     except Exception as e:
#         print(f"Error reading {file}: {e}")

# # Concatenate all valid DataFrames
# if dataframes:
#     combined_df = pd.concat(dataframes, ignore_index=True)
#     combined_df.to_csv("all_tables_combined.csv", index=False)
#     print(f"All tables combined into all_tables_combined.csv. Shape: {combined_df.shape}")
# else:
#     raise ValueError("No valid dataframes to concatenate.")


import os

print("Current Directory:", os.getcwd())
print("Files in Directory:", os.listdir())
