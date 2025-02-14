import os
import random
import pandas as pd


# Transformation functions

def unstack_dataframe(df):
    try:
        # Identify id columns based on name containing 'id'
        id_cols = [col for col in df.columns if 'id' in col.lower()]

        # Convert id columns to string, so they are treated as categorical
        for col in id_cols:
            df[col] = df[col].astype(str)

        non_numeric_cols = list(set(df.select_dtypes(include=['object']).columns.tolist() + id_cols))

        numeric_cols = [col for col in df.select_dtypes(exclude=['object']).columns.tolist() if col not in id_cols]

        if len(non_numeric_cols) < 2 or len(numeric_cols) == 0:
            raise ValueError(
                "Table must have at least two categorical columns and one numeric column (excluding id columns) for unstacking.")

        # Choose candidate columns for unstacking
        candidate_unstack_cols = [col for col in non_numeric_cols if col not in id_cols]
        if not candidate_unstack_cols:
            raise ValueError("No suitable candidate for unstacking found (all categorical columns are id columns).")

        # Select the column with the most repeated values -> (lowest ratio of unique values to total rows)
        unstack_col = min(candidate_unstack_cols, key=lambda col: df[col].nunique() / len(df))

        # All other categorical columns become the index columns
        index_cols = [col for col in non_numeric_cols if col != unstack_col]

        # Choose the first numeric column for pivoting, to preserve column names
        numeric_col = numeric_cols[0]

        # Group by the index columns plus the chosen unstack column, aggregating the chosen numeric column using mean
        grouped = df.groupby(index_cols + [unstack_col])[numeric_col].mean().reset_index()

        # Pivot the grouped data so that unique values of unstack_col become column names.
        df_unstacked = grouped.pivot(index=index_cols, columns=unstack_col, values=numeric_col)
        df_unstacked.columns = df_unstacked.columns.astype(str)  # ensure column names are strings
        df_unstacked = df_unstacked.reset_index()

        return df_unstacked

    except Exception as e:
        print(f"Error during unstacking: {e}")
        return df


def transpose_dataframe_inverse(df):
    try:
        print("\n Debug: Transposing DataFrame...")

        # Ensure the first column is treated as categorical. Rename columns if the first column isn't an object
        if df.dtypes[0] != 'object':
            df.columns = [f"Column_{i}" for i in range(df.shape[1])]

        transposed_df = df.set_index(df.columns[0]).T.reset_index()
        #print(f"Transposed DataFrame Shape: {transposed_df.shape}")
        return transposed_df

    except Exception as error:
        #print(f"Error during transpose inverse: {error}")
        return df


def apply_random_inverse_operator(df):
    non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()

    if len(non_numeric_cols) < 2 or len(numeric_cols) == 0:
        operator_name = "transpose"
    else:
        operator_name = random.choice(["unstack", "transpose"])

    if operator_name == "unstack":
        transformed_df = unstack_dataframe(df)
    else:
        transformed_df = transpose_dataframe_inverse(df)

    return transformed_df, operator_name


# Padding function

def pad_dataframe_to_101x50(df, target_rows=101, target_cols=50, fill_value=0):
    """
    Pads (and crops, if needed) the DataFrame so that it has exactly target_rows and target_cols.

    - If the DataFrame has fewer than target_cols columns, new columns are appended.
    - These new columns have header 0 and all their cells are filled with 0.
    - If it has fewer than target_rows rows, new rows are appended (all cells 0).
    - If the DataFrame is larger than the target size, it is cropped.
    """
    current_rows, current_cols = df.shape

    if current_cols < target_cols:

        num_new_cols = target_cols - current_cols

        pad_cols = pd.DataFrame(
            [[fill_value] * num_new_cols] * current_rows,
            index=df.index,
            columns=[0] * num_new_cols  # duplicate column name: 0
        )
        df = pd.concat([df, pad_cols], axis=1)

    # Pad rows
    if df.shape[0] < target_rows:
        num_new_rows = target_rows - df.shape[0]
        pad_rows = pd.DataFrame(
            [[fill_value] * df.shape[1]] * num_new_rows,
            columns=df.columns
        )
        df = pd.concat([df, pad_rows], axis=0, ignore_index=True)

    # Crop in case df is larger than target dimensions
    df = df.iloc[:target_rows, :target_cols]
    return df


# Processing & saving

def process_and_save_tables(input_folder, output_folder, mapping_filename="mapping.csv"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mapping = []

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".csv"):
            input_path = os.path.join(input_folder, filename)
            try:
                df = pd.read_csv(input_path)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            transformed_df, operator_name = apply_random_inverse_operator(df)

            padded_df = pad_dataframe_to_101x50(transformed_df, fill_value="")

            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_{operator_name}.csv"
            output_path = os.path.join(output_folder, output_filename)

            try:
                padded_df.to_csv(output_path, index=False)
                print(f"Processed {filename}: Applied '{operator_name}' transformation and padded to 101x50. Saved as {output_filename}.")
            except Exception as e:
                print(f"Error saving {output_filename}: {e}")

            # Save mapping information
            mapping.append({
                "original_file": filename,
                "transformation": operator_name,
                "output_file": output_filename
            })

    mapping_df = pd.DataFrame(mapping)
    mapping_path = os.path.join(output_folder, mapping_filename)
    try:
        mapping_df.to_csv(mapping_path, index=False)
        print(f"\nMapping saved to {mapping_path}")
    except Exception as e:
        print(f"Error saving mapping file: {e}")


# Example usage
if __name__ == "__main__":
    input_folder = "relational_tables"
    output_folder = "non_relational_tables"
    process_and_save_tables(input_folder, output_folder)
