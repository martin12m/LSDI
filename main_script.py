import os
import pandas as pd
import numpy as np
import random
import glob


# Path to the folder containing the CSV files
folder_path = "relational_tables_csv"

# Find all CSV files in the folder
csv_files = glob.glob(f"{folder_path}/*.csv")

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in the folder: {folder_path}")

# Combine all CSVs into one DataFrame
dataframes = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        if not df.empty:  # Ensure the file is not empty
            dataframes.append(df)
            print(f"{file} loaded successfully. Shape: {df.shape}")
        else:
            print(f"{file} is empty. Skipping.")
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Concatenate all valid DataFrames
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv("datasets/all_tables_combined.csv", index=False)
    print(f"All tables combined into all_tables_combined.csv. Shape: {combined_df.shape}")
else:
    raise ValueError("No valid dataframes to concatenate.")




def get_dataset_path():
    """Prompt the user for a dataset name and return the full path if it exists."""
    try:
        # Prompt for input
        dataset_name = input("Please enter the dataset name (including file extension, e.g., dataset.csv): ")
    except EOFError:
        # Fallback for environments that may not support input()
        print("Input prompt failed. Please enter the dataset name directly in the code.")
        dataset_name = "dataset1.csv"  # Set a default or modify this line

    datasets_folder = "datasets"  # Folder containing the datasets
    dataset_path = os.path.join(datasets_folder, dataset_name)

    if not os.path.exists(dataset_path):
        print(f"Error: The dataset '{dataset_name}' was not found in the '{datasets_folder}' folder.")
        return None
    
    return dataset_path

def load_dataset(dataset_path: str):
    """Load a dataset from a given path using Pandas."""
    try:
        df = pd.read_csv(dataset_path)
        print(f"Successfully loaded dataset from {dataset_path}.")
        return df
    except Exception as e:
        print(f"Failed to load the dataset: {e}")
        return None

# Example inverse transformations
def unstack_table(df: pd.DataFrame) -> pd.DataFrame:
    """Inverse operation of 'stack', creating a non-relational form by spreading values."""
    if len(df.columns) < 2:
        return df  # Not enough columns to unstack
    unstacked_df = df.copy()
    stacked_columns = df.columns[1:]  # Select columns to "unstack"
    unstacked_df = unstacked_df.pivot_table(index=df.columns[0], columns=stacked_columns)
    unstacked_df.columns = ['_'.join(map(str, col)).strip() for col in unstacked_df.columns.values]
    return unstacked_df.reset_index()

def transpose_table_inverse(df: pd.DataFrame) -> pd.DataFrame:
    """Inverse of a transpose operation - transposing again restores original form."""
    return df.transpose().reset_index()

def wide_to_long_inverse(df: pd.DataFrame) -> pd.DataFrame:
    """Inverse of 'wide-to-long', splitting column groups into separate columns."""
    if len(df.columns) < 4:
        return df  # Not enough columns to create groups
    df_copy = df.copy()
    group_size = random.randint(2, len(df.columns) // 2)
    num_groups = len(df.columns) // group_size
    new_columns = [f"Group_{i}_{j}" for i in range(num_groups) for j in range(group_size)]
    df_copy.columns = new_columns[:len(df_copy.columns)]  # Rename columns
    return df_copy

def pivot_table_inverse(df: pd.DataFrame) -> pd.DataFrame:
    """Inverse of 'pivot' operation - replicates row groups."""
    if len(df.columns) < 2:
        return df  # Not enough columns to pivot
    df["GroupID"] = df.index % random.randint(2, min(len(df), 5))  # Create a repeating group identifier
    pivoted_df = df.pivot(index="GroupID", columns=df.columns[0], values=df.columns[1:])
    return pivoted_df.reset_index()

def generate_non_relational_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a random inverse transformation to generate a non-relational table."""
    inverse_transformations = [unstack_table, transpose_table_inverse, wide_to_long_inverse, pivot_table_inverse]
    chosen_transform = random.choice(inverse_transformations)
    
    try:
        transformed_df = chosen_transform(df)
    except Exception as e:
        print(f"Inverse transformation failed: {e}. Returning original table.")
        return df
    
    return transformed_df

def save_non_relational_table(df: pd.DataFrame, output_name: str):
    """Save the non-relational table to a CSV file."""
    output_folder = "transformed_datasets"
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    output_path = os.path.join(output_folder, output_name)
    df.to_csv(output_path, index=False)
    print(f"Non-relational table saved to {output_path}")

def main():
    dataset_path = get_dataset_path()
    if not dataset_path:
        return
    
    df = load_dataset(dataset_path)
    if df is None:
        return
    
    # Generate a single non-relational table as an example
    non_relational_table = generate_non_relational_table(df)
    print("Transformed non-relational table preview:")
    print(non_relational_table.head())

    # Save the transformed table
    output_name = "non_relational_" + os.path.basename(dataset_path)
    save_non_relational_table(non_relational_table, output_name)
    
if __name__ == "__main__":
    main()


