# import os
# import pandas as pd
# import numpy as np
# import random

# def get_dataset_path():
#     """Prompt the user for a dataset name and return the full path if it exists."""
#     try:
#         # Prompt for input
#         dataset_name = input("Please enter the dataset name (including file extension, e.g., dataset.csv): ")
#     except EOFError:
#         # Fallback for environments that may not support input()
#         print("Input prompt failed. Please enter the dataset name directly in the code.")
#         dataset_name = "dataset1.csv"  # Set a default or modify this line

#     datasets_folder = "datasets"  # Folder containing the datasets
#     dataset_path = os.path.join(datasets_folder, dataset_name)

#     if not os.path.exists(dataset_path):
#         print(f"Error: The dataset '{dataset_name}' was not found in the '{datasets_folder}' folder.")
#         return None
    
#     return dataset_path

# def load_dataset(dataset_path: str):
#     """Load a dataset from a given path using Pandas."""
#     try:
#         df = pd.read_csv(dataset_path)
#         print(f"Successfully loaded dataset from {dataset_path}.")
#         return df
#     except Exception as e:
#         print(f"Failed to load the dataset: {e}")
#         return None

# # Example inverse transformations
# def unstack_table(df: pd.DataFrame) -> pd.DataFrame:
#     """Inverse operation of 'stack', creating a non-relational form by spreading values."""
#     if len(df.columns) < 2:
#         return df  # Not enough columns to unstack
#     unstacked_df = df.copy()
#     stacked_columns = df.columns[1:]  # Select columns to "unstack"
#     unstacked_df = unstacked_df.pivot_table(index=df.columns[0], columns=stacked_columns)
#     unstacked_df.columns = ['_'.join(map(str, col)).strip() for col in unstacked_df.columns.values]
#     return unstacked_df.reset_index()

# def transpose_table_inverse(df: pd.DataFrame) -> pd.DataFrame:
#     """Inverse of a transpose operation - transposing again restores original form."""
#     return df.transpose().reset_index()

# def wide_to_long_inverse(df: pd.DataFrame) -> pd.DataFrame:
#     """Inverse of 'wide-to-long', splitting column groups into separate columns."""
#     if len(df.columns) < 4:
#         return df  # Not enough columns to create groups
#     df_copy = df.copy()
#     group_size = random.randint(2, len(df.columns) // 2)
#     num_groups = len(df.columns) // group_size
#     new_columns = [f"Group_{i}_{j}" for i in range(num_groups) for j in range(group_size)]
#     df_copy.columns = new_columns[:len(df_copy.columns)]  # Rename columns
#     return df_copy

# def pivot_table_inverse(df: pd.DataFrame) -> pd.DataFrame:
#     """Inverse of 'pivot' operation - replicates row groups."""
#     if len(df.columns) < 2:
#         return df  # Not enough columns to pivot
#     df["GroupID"] = df.index % random.randint(2, min(len(df), 5))  # Create a repeating group identifier
#     pivoted_df = df.pivot(index="GroupID", columns=df.columns[0], values=df.columns[1:])
#     return pivoted_df.reset_index()

# def generate_non_relational_table(df: pd.DataFrame) -> pd.DataFrame:
#     """Apply a random inverse transformation to generate a non-relational table."""
#     inverse_transformations = [unstack_table, transpose_table_inverse, wide_to_long_inverse, pivot_table_inverse]
#     chosen_transform = random.choice(inverse_transformations)
    
#     try:
#         transformed_df = chosen_transform(df)
#     except Exception as e:
#         print(f"Inverse transformation failed: {e}. Returning original table.")
#         return df
    
#     return transformed_df

# def save_non_relational_table(df: pd.DataFrame, output_name: str):
#     """Save the non-relational table to a CSV file."""
#     output_folder = "transformed_datasets"
#     os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
#     output_path = os.path.join(output_folder, output_name)
#     df.to_csv(output_path, index=False)
#     print(f"Non-relational table saved to {output_path}")

# def main():
#     dataset_path = get_dataset_path()
#     if not dataset_path:
#         return
    
#     df = load_dataset(dataset_path)
#     if df is None:
#         return
    
#     # Generate a single non-relational table as an example
#     non_relational_table = generate_non_relational_table(df)
#     print("Transformed non-relational table preview:")
#     print(non_relational_table.head())

#     # Save the transformed table
#     output_name = "non_relational_" + os.path.basename(dataset_path)
#     save_non_relational_table(non_relational_table, output_name)
    
# if __name__ == "__main__":
#     main()



import os
import pandas as pd
import numpy as np
import random

def prompt_for_dataset():
    """Ask the user for a dataset filename and return its full path if it exists."""
    dataset_name = input("Enter the dataset filename (with extension, e.g., dataset.csv): ")
    datasets_directory = "datasets"  # Directory where datasets are stored
    full_path = os.path.join(datasets_directory, dataset_name)

    if not os.path.isfile(full_path):
        print(f"Error: The dataset '{dataset_name}' could not be found in the '{datasets_directory}' directory.")
        return None
    
    return full_path

def read_dataset(file_path: str):
    """Load a dataset from the specified file path using Pandas."""
    try:
        data_frame = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}.")
        return data_frame
    except Exception as error:
        print(f"Error loading dataset: {error}")
        return None

# Inverse transformation functions
def reverse_stack(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse the 'stack' operation to create a non-relational format."""
    if df.shape[1] < 2:
        return df  # Not enough columns to reverse stack
    reshaped_df = df.copy()
    columns_to_expand = df.columns[1:]
    reshaped_df = reshaped_df.pivot_table(index=df.columns[0], columns=columns_to_expand)
    reshaped_df.columns = ['_'.join(map(str, col)).strip() for col in reshaped_df.columns.values]
    return reshaped_df.reset_index()

def reverse_transpose(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse the transpose operation to restore the original format."""
    return df.transpose().reset_index()

def reverse_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse the 'wide-to-long' transformation by separating column groups."""
    if df.shape[1] < 4:
        return df  # Not enough columns to create groups
    df_copy = df.copy()
    group_size = random.randint(2, df.shape[1] // 2)
    group_count = df.shape[1] // group_size
    new_column_names = [f"Group_{i}_{j}" for i in range(group_count) for j in range(group_size)]
    df_copy.columns = new_column_names[:df_copy.shape[1]]  # Rename columns
    return df_copy

def reverse_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse the 'pivot' operation to replicate row groups."""
    if df.shape[1] < 2:
        return df  # Not enough columns to pivot
    df["GroupID"] = df.index % random.randint(2, min(len(df), 5))  # Create a repeating identifier for groups
    pivoted_df = df.pivot(index="GroupID", columns=df.columns[0], values=df.iloc[:, 1:])
    return pivoted_df.reset_index()

def create_non_relational_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a random inverse transformation to create a non-relational table."""
    transformations = [reverse_stack, reverse_transpose, reverse_wide_to_long, reverse_pivot]
    selected_transformation = random.choice(transformations)
    
    try:
        result_df = selected_transformation(df)
    except Exception as error:
        print(f"Failed to apply inverse transformation: {error}. Returning the original table.")
        return df
    
    return result_df

def export_non_relational_table(df: pd.DataFrame, filename: str):
    """Save the non-relational table to a CSV file."""
    output_dir = "transformed_datasets"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    output_file_path = os.path.join(output_dir, filename)
    df.to_csv(output_file_path, index=False)
    print(f"Non-relational table has been saved to {output_file_path}")

def execute_script():
    dataset_path = prompt_for_dataset()
    if not dataset_path:
        return
    
    df = read_dataset(dataset_path)
    if df is None:
        return
    
    # Create a non-relational table as an example
    non_relational_table = create_non_relational_table(df)
    print("Preview of the transformed non-relational table:")
    print(non_relational_table.head())

    # Save the transformed table
    output_filename = "non_relational_" + os.path.basename(dataset_path)
    export_non_relational_table(non_relational_table, output_filename)

if __name__ == "__main__":
    execute_script()
