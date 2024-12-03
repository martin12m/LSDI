# import os
# import pandas as pd
# import numpy as np
# import random
# import glob
# # import FakeTables as FakeTables
# # import combine_all_files as combine_all_files



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
#     dataset_path = os.path.join( dataset_name)

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
    
#     if __name__ == "__main__":
#         main()


'''
import os
import pandas as pd
import random

# Transformation Functions
def unstack_table(df):
    """Convert columns to rows, making the table non-relational."""
    if len(df.columns) < 2:
        return df  # Skip if not enough columns
    return pd.melt(df, id_vars=df.columns[0], var_name="Attributes", value_name="Values")

def transpose_table_inverse(df):
    """Transpose rows to columns (reversible operation)."""
    return df.T.reset_index()

def wide_to_long_inverse(df):
    """Randomly group columns into a long format."""
    if len(df.columns) < 4:
        return df  # Skip if not enough columns
    group_size = random.randint(2, max(2, len(df.columns) // 2))
    long_df = pd.melt(df, id_vars=df.columns[:group_size], var_name="Attributes", value_name="Values")
    return long_df

def pivot_table_inverse(df):
    """Simulate pivoting groups of rows."""
    if len(df.columns) < 2:
        return df  # Skip if not enough columns
    try:
        # Handle duplicates by adding a unique group ID
        if df.duplicated(subset=df.columns[0]).any():
            df["UniqueID"] = df.groupby(df.columns[0]).cumcount()

        # Pivot the table
        pivoted_df = df.pivot(
            index=["Group", "UniqueID"] if "UniqueID" in df.columns else "Group",
            columns=df.columns[0],
            values=df.columns[1:]
        ).reset_index()

        # Drop UniqueID if it was added
        if "UniqueID" in pivoted_df.columns:
            pivoted_df = pivoted_df.drop(columns=["UniqueID"])

        return pivoted_df
    except Exception as e:
        print(f"Failed to pivot table: {e}")
        return df

# Main function to apply transformations
def generate_non_relational_table(df, table_id, transformation_log):
    """Apply a random transformation and log the operation."""
    transformations = {
        "unstack": unstack_table,
        "transpose": transpose_table_inverse,
        "wide_to_long": wide_to_long_inverse,
        "pivot": pivot_table_inverse,
    }
    transform_name, transform_func = random.choice(list(transformations.items()))
    try:
        transformed_df = transform_func(df)
        transformation_log[table_id] = transform_name
        return transformed_df
    except Exception as e:
        print(f"Transformation failed for {table_id}: {e}")
        transformation_log[table_id] = "none"
        return df  # Return original table if transformation fails

# Process all relational tables
def process_all_relational_tables():
    input_folder = "relational_tables_csv"  # Folder with relational tables
    output_folder = "transformed_datasets"  # Folder to save non-relational tables
    os.makedirs(output_folder, exist_ok=True)

    transformation_log = {}
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            table_id = os.path.splitext(file)[0]  # Use the filename as table ID
            try:
                df = pd.read_csv(os.path.join(input_folder, file))
                transformed_df = generate_non_relational_table(df, table_id, transformation_log)
                transformed_df.to_csv(os.path.join(output_folder, f"non_relational_{file}"), index=False)
                print(f"Processed and saved: {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # Save transformation log to labels.csv
    labels_path = "labels.csv"
    pd.DataFrame(list(transformation_log.items()), columns=["table_id", "transformation"]).to_csv(labels_path, index=False)
    print(f"Transformation log saved to {labels_path}")

process_all_relational_tables()
'''

#MAIN_SCRIPT_A_UNSTACK_CORRECTED


import os
import pandas as pd
import random

# Transformation Functions
def unstack_table(df):
    """Convert columns to rows without losing any information."""
    if len(df.columns) < 2:
        return df  # Not enough columns to melt
    
    # Add a unique identifier for each row
    df = df.reset_index().rename(columns={'index': 'RowID'})
    
    # Melt the DataFrame, keeping 'RowID' to preserve all data
    melted_df = pd.melt(
        df,
        id_vars=['RowID'],
        var_name='Attributes',
        value_name='Values'
    )
    
    return melted_df

def transpose_table_inverse(df):
    """Transpose rows to columns (reversible operation)."""
    return df.T.reset_index()

def wide_to_long_inverse(df):
    """Randomly group columns into a long format."""
    if len(df.columns) < 4:
        return df  # Skip if not enough columns
    group_size = random.randint(2, max(2, len(df.columns) // 2))
    long_df = pd.melt(df, id_vars=df.columns[:group_size], var_name="Attributes", value_name="Values")
    return long_df

def pivot_table_inverse(df):
    """Simulate pivoting groups of rows."""
    if len(df.columns) < 2:
        return df  # Skip if not enough columns
    try:
        # Handle duplicates by adding a unique group ID
        if df.duplicated(subset=df.columns[0]).any():
            df["UniqueID"] = df.groupby(df.columns[0]).cumcount()

        # Pivot the table
        pivoted_df = df.pivot(
            index=["Group", "UniqueID"] if "UniqueID" in df.columns else "Group",
            columns=df.columns[0],
            values=df.columns[1:]
        ).reset_index()

        # Drop UniqueID if it was added
        if "UniqueID" in pivoted_df.columns:
            pivoted_df = pivoted_df.drop(columns=["UniqueID"])

        return pivoted_df
    except Exception as e:
        print(f"Failed to pivot table: {e}")
        return df

# Main function to apply transformations
def generate_non_relational_table(df, table_id, transformation_log):
    """Apply a random transformation and log the operation."""
    transformations = {
        "unstack": unstack_table,
        "transpose": transpose_table_inverse,
        "wide_to_long": wide_to_long_inverse,
        "pivot": pivot_table_inverse,
    }
    transform_name, transform_func = random.choice(list(transformations.items()))
    try:
        transformed_df = transform_func(df)
        transformation_log[table_id] = transform_name
        return transformed_df
    except Exception as e:
        print(f"Transformation failed for {table_id}: {e}")
        transformation_log[table_id] = "none"
        return df  # Return original table if transformation fails

# Process all relational tables
def process_all_relational_tables():
    input_folder = "relational_tables_csv"  # Folder with relational tables
    output_folder = "transformed_datasets"  # Folder to save non-relational tables
    os.makedirs(output_folder, exist_ok=True)

    transformation_log = {}
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            table_id = os.path.splitext(file)[0]  # Use the filename as table ID
            try:
                df = pd.read_csv(os.path.join(input_folder, file))
                transformed_df = generate_non_relational_table(df, table_id, transformation_log)
                transformed_df.to_csv(os.path.join(output_folder, f"non_relational_{file}"), index=False)
                print(f"Processed and saved: {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # Save transformation log to labels.csv
    labels_path = "labels.csv"
    pd.DataFrame(list(transformation_log.items()), columns=["table_id", "transformation"]).to_csv(labels_path, index=False)
    print(f"Transformation log saved to {labels_path}")

process_all_relational_tables()


