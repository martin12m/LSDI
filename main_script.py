#MAIN_SCRIPT_A_UNSTACK_CORRECTED


import os
import pandas as pd
import random
import creates_Relational_tables as create_relational_tables


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


