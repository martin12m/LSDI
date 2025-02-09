import pandas as pd
import numpy as np
import random
import os

# ---- Operators and Inverse Operators ----
def unstack_dataframe(df):
    try:
        non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()

        if len(non_numeric_cols) < 2 or len(numeric_cols) == 0:
            raise ValueError("Table must contain categorical and numeric columns for unstacking.")
        
        index_cols = non_numeric_cols[:-1]
        unstack_col = non_numeric_cols[-1]

        unstacked_df = df.pivot(index=index_cols, columns=unstack_col, values=numeric_cols[0])
        unstacked_df = unstacked_df.reset_index()

        return unstacked_df

    except Exception as e:
        print(f"Error during unstacking: {e}")
        return df

def transpose_dataframe_inverse(df):
    try:
        print("\n🛠 Debug: Transposing DataFrame...")

        # Ensure first column is categorical (Avoid numeric index issues)
        if df.dtypes[0] != 'object':
            df.columns = [f"Column_{i}" for i in range(df.shape[1])]

        # Transpose and reset index
        transposed_df = df.set_index(df.columns[0]).T.reset_index()

        # Print debug info
        print(f"✅ Transposed DataFrame Shape: {transposed_df.shape}")
        return transposed_df

    except Exception as error:
        print(f"⚠ Error during transpose inverse: {error}")
        return df

def apply_random_inverse_operator(df):
    operators = {
        "unstack": unstack_dataframe,
        "transpose": transpose_dataframe_inverse
    }
    operator_name = random.choice(list(operators.keys()))
    transformed_df = operators[operator_name](df)
    return transformed_df, operator_name

def resize_dataframe(df, target_rows=101, target_cols=50):
    try:
        print("\n🛠 Debug: Resizing DataFrame...")

        # Convert all values to numeric, replacing non-numeric with NaN
        df_numeric = df.apply(pd.to_numeric, errors='coerce')

        # Flatten values and remove NaNs
        values = df_numeric.values.flatten()
        values = values[~np.isnan(values)]  # Ensure only numeric values remain

        total_values = target_rows * target_cols

        # Adjust number of values dynamically
        if len(values) > total_values:
            # Truncate excess values
            values = values[:total_values]
        elif len(values) < total_values:
            # Pad with repeated values
            extra_needed = total_values - len(values)
            repeated_values = np.tile(values, (extra_needed // len(values)) + 1)[:extra_needed]
            values = np.concatenate([values, repeated_values])

        # Create resized dataframe
        resized_df = pd.DataFrame(values.reshape(target_rows, target_cols))

        # Maintain original column names if possible
        if len(df.columns) >= target_cols:
            resized_df.columns = df.columns[:target_cols]
        else:
            # Add placeholder column names
            new_columns = list(df.columns) + [f"Extra_Feature_{i+1}" for i in range(target_cols - len(df.columns))]
            resized_df.columns = new_columns

        print(f"✅ Resized DataFrame Shape: {resized_df.shape}")
        return resized_df

    except Exception as e:
        print(f"⚠ Error during resizing: {e}")
        return df

def process_files(input_folder, output_folder):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all CSV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {file_path}")

            try:
                # Load the CSV file with specified encoding
                relational_data = pd.read_csv(file_path, encoding='latin1')  # or use 'ISO-8859-1'

                # Apply a random inverse operator to the relational table
                non_relational_table, operator_name = apply_random_inverse_operator(relational_data)
                print(f"Applied random inverse operator: {operator_name}")

                if non_relational_table.empty:
                    print("Transformed table is empty. Skipping.")
                    continue

                resized_df = resize_dataframe(non_relational_table)
                resized_file_path = os.path.join(output_folder, f"resized_{filename}")
                resized_df.to_csv(resized_file_path, index=False)
                print(f"Saved resized table to: {resized_file_path}")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

def main():
    input_folder = "kaggle_files"
    output_folder = "resized_kaggle_files"
    process_files(input_folder, output_folder)

if __name__ == "__main__":
    main()