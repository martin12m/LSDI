# #MAIN_SCRIPT_A_UNSTACK_CORRECTED


# import os
# import pandas as pd
# import random

# def unstack_table(df):
#     if len(df.columns) < 2:
#         return df 
    
#     df = df.reset_index().rename(columns={'index': 'RowID'})
    
#     melted_df = pd.melt(
#         df,
#         id_vars=['RowID'],
#         var_name='Attributes',
#         value_name='Values'
#     )
    
#     return melted_df

# def transpose_table_inverse(df):
#     return df.T.reset_index()

# def wide_to_long_inverse(df):
#     if len(df.columns) < 4:
#         return df 
#     group_size = random.randint(2, max(2, len(df.columns) // 2))
#     long_df = pd.melt(df, id_vars=df.columns[:group_size], var_name="Attributes", value_name="Values")
#     return long_df

# def pivot_table_inverse(df):
#     if len(df.columns) < 2:
#         return df 
#     try:
        
#         if df.duplicated(subset=df.columns[0]).any():
#             df["UniqueID"] = df.groupby(df.columns[0]).cumcount()

#         pivoted_df = df.pivot(
#             index=["Group", "UniqueID"] if "UniqueID" in df.columns else "Group",
#             columns=df.columns[0],
#             values=df.columns[1:]
#         ).reset_index()

#         if "UniqueID" in pivoted_df.columns:
#             pivoted_df = pivoted_df.drop(columns=["UniqueID"])

#         return pivoted_df
#     except Exception as e:
#         print(f"Failed to pivot table: {e}")
#         return df


# def generate_non_relational_table(df, table_id, transformation_log):
#     transformations = {
#         "unstack": unstack_table,
#         "transpose": transpose_table_inverse,
#         "wide_to_long": wide_to_long_inverse,
#         "pivot": pivot_table_inverse,
#     }
#     transform_name, transform_func = random.choice(list(transformations.items()))
#     try:
#         transformed_df = transform_func(df)
#         transformation_log[table_id] = transform_name
#         return transformed_df
#     except Exception as e:
#         print(f"Transformation failed for {table_id}: {e}")
#         transformation_log[table_id] = "none"
#         return df  # Return original table if transformation fails


# def process_all_relational_tables():
#     input_folder = "relational_tables"  
#     output_folder = "transformed_datasets"  
#     os.makedirs(output_folder, exist_ok=True)

#     transformation_log = {}
#     for file in os.listdir(input_folder):
#         if file.endswith(".csv"):
#             table_id = os.path.splitext(file)[0]  # Use the filename as table ID
#             try:
#                 df = pd.read_csv(os.path.join(input_folder, file))
#                 transformed_df = generate_non_relational_table(df, table_id, transformation_log)
#                 transformed_df.to_csv(os.path.join(output_folder, f"non_relational_{file}"), index=False)
#                 print(f"Processed and saved: {file}")
#             except Exception as e:
#                 print(f"Error processing {file}: {e}")
    

#     labels_path = "labelsTEST.csv"
#     pd.DataFrame(list(transformation_log.items()), columns=["table_id", "transformation"]).to_csv(labels_path, index=False)
#     print(f"Transformation log saved to {labels_path}")

# process_all_relational_tables()


# create_Non_Relational_tables.py

# create_Non_Relational_tables.py

import os
import pandas as pd
import random

def unstack_dataframe(dataframe):
    if dataframe.shape[1] < 2:
        return dataframe
    
    dataframe = dataframe.reset_index()
    dataframe.rename(columns={'index': 'RowID'}, inplace=True)
    melted = pd.melt(dataframe, id_vars=['RowID'], var_name='Attributes', value_name='Values')
    return melted

def transpose_dataframe(dataframe):
    transposed = dataframe.T
    transposed.reset_index(inplace=True)
    return transposed

def convert_wide_to_long(dataframe):
    if dataframe.shape[1] < 4:
        return dataframe
    
    group_size = random.randint(2, max(2, dataframe.shape[1] // 2))
    long_format = pd.melt(dataframe, id_vars=dataframe.columns[:group_size], var_name="Attributes", value_name="Values")
    return long_format

def pivot_dataframe(dataframe):
    if dataframe.shape[1] < 2:
        return dataframe
    
    try:
        if dataframe.duplicated(subset=dataframe.columns[0]).any():
            dataframe["UniqueID"] = dataframe.groupby(dataframe.columns[0]).cumcount()

        index_columns = ["Group", "UniqueID"] if "UniqueID" in dataframe.columns else "Group"
        pivoted = dataframe.pivot(index=index_columns, columns=dataframe.columns[0], values=dataframe.columns[1:])
        pivoted.reset_index(inplace=True)

        if "UniqueID" in pivoted.columns:
            pivoted.drop(columns=["UniqueID"], inplace=True)

        return pivoted
    except Exception as error:
        print(f"Error during pivoting: {error}")
        return dataframe

def create_non_relational_table(dataframe, table_identifier, log):
    transformation_methods = {
        "unstack": unstack_dataframe,
        "transpose": transpose_dataframe,
        "wide_to_long": convert_wide_to_long,
        "pivot": pivot_dataframe,
    }
    
    transformation_name, transformation_function = random.choice(list(transformation_methods.items()))
    
    try:
        transformed_data = transformation_function(dataframe)
        log[table_identifier] = transformation_name
        return transformed_data
    except Exception as error:
        print(f"Failed to transform table {table_identifier}: {error}")
        log[table_identifier] = "none"
        return dataframe

def process_relational_tables():
    input_directory = "relational_tables"
    output_directory = "transformed_datasets"
    os.makedirs(output_directory, exist_ok=True)

    transformation_log = {}
    
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            table_id = os.path.splitext(filename)[0]
            try:
                df = pd.read_csv(os.path.join(input_directory, filename))
                transformed_df = create_non_relational_table(df, table_id, transformation_log)
                transformed_df.to_csv(os.path.join(output_directory, f"non_relational_{filename}"), index=False)
                print(f"Successfully processed and saved: {filename}")
            except Exception as error:
                print(f"Error while processing {filename}: {error}")

    log_file_path = "labelsTEST.csv"
    pd.DataFrame(list(transformation_log.items()), columns=["table_id", "transformation"]).to_csv(log_file_path, index=False)
    print(f"Transformation log has been saved to {log_file_path}")

process_relational_tables()