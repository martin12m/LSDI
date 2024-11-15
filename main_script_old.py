import os
import pandas as pd
import numpy as np
import random

def get_dataset_path():
    """Prompt the user for a dataset name and return the full path if it exists."""
    dataset_name = input("Please enter the dataset name (including file extension, e.g., dataset.csv): ")
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

def generate_dataset(dataset: list[pd.DataFrame], num_samples: int = 5000) -> list[pd.DataFrame]:
    """Generate a dataset of non-relational tables using inverse transformations."""
    generated_dataset = []
    for i, df in enumerate(dataset):
        for _ in range(num_samples // len(dataset)):
            non_relational_df = generate_non_relational_table(df)
            generated_dataset.append(non_relational_df)
    return generated_dataset

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
    
    
if __name__ == "__main__":
    main()

