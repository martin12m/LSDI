# import os
# import pandas as pd
# import numpy as np
# import random
# from faker import Faker

# # Configuration parameters
# TOTAL_TABLES = 5000
# ROW_COUNT = 100
# COLUMN_COUNT = 50
# FOREIGN_KEY_PROBABILITY = 0.1

# # Initialize Faker for generating realistic data
# data_generator = Faker()

# # Define the structure of predefined columns
# PREDEFINED_COLUMNS = {
#     'id': 'int',
#     'name': 'str',
#     'age': 'int',
#     'email': 'str',
#     'address': 'str',
#     'phone': 'str',
#     'created_at': 'datetime',
#     'status': 'str',
#     'score': 'float',
#     'category': 'str',
#     'price': 'float',
#     'quantity': 'int',
#     'description': 'str',
#     'active': 'bool',
#     'rating': 'float'
# }

# # Dynamically add additional columns
# for i in range(15, COLUMN_COUNT):
#     PREDEFINED_COLUMNS[f'custom_col_{i}'] = random.choice(['int', 'float', 'str'])

# # Ensure COLUMN_COUNT does not exceed the available columns
# assert COLUMN_COUNT <= len(PREDEFINED_COLUMNS), "COLUMN_COUNT exceeds predefined columns."

# def generate_data(column_name, data_type, row_count):
#     if data_type == 'int':
#         return np.random.randint(1, 1000, size=row_count)
#     elif data_type == 'float':
#         return np.random.uniform(1.0, 1000.0, size=row_count).round(2)
#     elif data_type == 'str':
#         return generate_string_data(column_name, row_count)
#     elif data_type == 'datetime':
#         return [data_generator.date_time_between(start_date='-2y', end_date='now') for _ in range(row_count)]
#     elif data_type == 'bool':
#         return [random.choice([True, False]) for _ in range(row_count)]
#     else:
#         return [None] * row_count  # Fallback

# def generate_string_data(column_name, row_count):
#     if column_name == 'email':
#         return [data_generator.email() for _ in range(row_count)]
#     # Additional string cases (no changes)

# # Create and save relational tables
# output_directory = 'relational_tables_csv'
# os.makedirs(output_directory, exist_ok=True)
# tables_dict = {}
# for table_index in range(1, TOTAL_TABLES + 1):
#     table_name = f'table_{table_index}'
#     try:
#         selected_columns = random.sample(list(PREDEFINED_COLUMNS.keys()), COLUMN_COUNT - 1)
#         selected_columns.insert(0, 'id')  # Ensure 'id' is included

#         table_content = {}
#         for column in selected_columns:
#             if random.random() < FOREIGN_KEY_PROBABILITY and table_index > 1:
#                 foreign_table = random.choice(list(tables_dict.keys()))
#                 foreign_ids = tables_dict[foreign_table]['id'].tolist()
#                 table_content[column] = np.random.choice(foreign_ids, size=ROW_COUNT) if foreign_ids else \
#                     np.random.randint(1, 1000, size=ROW_COUNT)
#             else:
#                 table_content[column] = generate_data(column, PREDEFINED_COLUMNS[column], ROW_COUNT)

#         df = pd.DataFrame(table_content)
#         tables_dict[table_name] = df
#         csv_file_path = os.path.join(output_directory, f"{table_name}.csv")
#         df.to_csv(csv_file_path, index=False)

#     except Exception as e:
#         print(f"Error generating table {table_name}: {e}")
#         continue  # Skip failed table generation

#     if table_index % 1000 == 0:
#         print(f"Generated {table_index} tables")
        
        
        
        
        
        
        
        

# import os
# import pandas as pd
# import numpy as np
# import random

# # Directory paths
# RELATIONAL_DIR = 'relational_tables_csv'
# NON_RELATIONAL_DIR = 'non_relational_tables_csv'
# os.makedirs(NON_RELATIONAL_DIR, exist_ok=True)

# # Log file to track transformations
# LOG_FILE = 'labels.csv'

# # List of transformations
# TRANSFORMATIONS = ["unstack", "transpose", "wide_to_long",]

# def unstack_table(df):
#     """Convert a DataFrame using unstack operation."""
#     return df.stack().reset_index(name='value')

# def transpose_table_inverse(df):
#     """Transpose the table."""
#     return df.T.reset_index()

# def wide_to_long_inverse(df):
#     """Transform a wide table into long format."""
#     id_vars = df.columns[:2] if len(df.columns) > 1 else df.columns[:1]
#     value_vars = df.columns[2:] if len(df.columns) > 2 else df.columns[1:]
#     return pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='attribute', value_name='value')

# # def pivot_table_inverse(df):
# #     """Apply pivot table transformation."""
# #     if len(df.columns) < 3:
# #         raise ValueError("Not enough columns to apply pivot.")
# #     return df.pivot(index=df.columns[0], columns=df.columns[1], values=df.columns[2]).reset_index()

# def apply_transformation(df, transformation_type):
#     """Apply a specific transformation to the DataFrame."""
#     if transformation_type == "unstack":
#         return unstack_table(df)
#     elif transformation_type == "transpose":
#         return transpose_table_inverse(df)
#     elif transformation_type == "wide_to_long":
#         return wide_to_long_inverse(df)
#     # elif transformation_type == "pivot":
#     #     return pivot_table_inverse(df)
#     else:
#         raise ValueError(f"Unknown transformation type: {transformation_type}")

# def process_relational_table(file_path, table_id, log):
#     """Process a single relational table and apply a random transformation."""
#     try:
#         # Load the relational table
#         df = pd.read_csv(file_path)

#         # Choose a random transformation
#         transformation_type = random.choice(TRANSFORMATIONS)
#         print(f"Applying transformation '{transformation_type}' on {file_path}")

#         # Apply transformation
#         transformed_df = apply_transformation(df, transformation_type)

#         # Save transformed table
#         output_path = os.path.join(NON_RELATIONAL_DIR, f"{table_id}_non_relational.csv")
#         transformed_df.to_csv(output_path, index=False)

#         # Log transformation
#         log.append({"table_id": table_id, "transformation": transformation_type})
    
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")
#         # Log failed transformation
#         log.append({"table_id": table_id, "transformation": "none"})

# def process_all_relational_tables():
#     """Process all relational tables in the directory."""
#     log = []

#     # Iterate through all CSV files in the relational directory
#     for file_name in os.listdir(RELATIONAL_DIR):
#         if file_name.endswith('.csv'):
#             file_path = os.path.join(RELATIONAL_DIR, file_name)
#             table_id = os.path.splitext(file_name)[0]
#             process_relational_table(file_path, table_id, log)

#     # Save log file
#     log_df = pd.DataFrame(log)
#     log_df.to_csv(LOG_FILE, index=False)
#     print(f"Transformation log saved to {LOG_FILE}")

# if __name__ == "__main__":
#     process_all_relational_tables()
        
        
        

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Set folder path containing 5000 CSV files
folder_path = "relational_tables_csv"

def load_and_merge_data(folder_path):
    """
    Load all CSV files from the folder, process, and merge into a single dataset.
    """
    all_data = []  # List to store data from all files
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Loading file: {file_path}")
            
            # Load the CSV
            df = pd.read_csv(file_path)
            
            # Example: Basic preprocessing (normalize, fill NaNs, encode labels)
            df.fillna(0, inplace=True)  # Replace missing values with 0
            df = df.select_dtypes(include=[np.number])  # Keep numeric columns only
            
            # Add data to the list
            all_data.append(df)
    
    # Concatenate all data into a single DataFrame
    merged_data = pd.concat(all_data, ignore_index=True)
    print(f"Merged dataset shape: {merged_data.shape}")
    print(f"Column names in merged dataset: {list(merged_data.columns)}")
    merged_data.to_csv("merged_data.csv", index=False)  # Specify the desired file name
    return merged_data

def preprocess_data(data, target_column):
    """
    Preprocess merged data: Separate features and labels, normalize data, and reshape.
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset. Available columns: {list(data.columns)}")
    
    # Separate features (X) and labels (y)
    X = data.drop(columns=[target_column]).values  # Features
    y = data[target_column].values  # Labels
    
    # Normalize features
    X = X / np.max(X)  # Scale values to [0, 1]
    
    # Reshape features for CNN (example: 64x64 grid)
    try:
        X = X.reshape(-1, 64, 64, 1)  # Adjust dimensions based on your dataset
    except ValueError as e:
        print("Error in reshaping the data. Ensure the feature size matches 64x64. Check input data shape.")
        raise e
    
    # One-hot encode labels
    num_classes = len(np.unique(y))
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    
    return X, y, num_classes

def build_model(input_shape, num_classes):
    """
    Build a CNN model for classification tasks.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the CNN model with the training data.
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32
    )
    return history

if __name__ == "__main__":
    # Step 1: Load and merge data from CSV files
    data = load_and_merge_data(folder_path)
    
    # Step 2: Preprocess the data
    target_column = "label"  # Replace with your actual label column name
    try:
        X, y, num_classes = preprocess_data(data, target_column)
    except ValueError as e:
        print(f"Preprocessing error: {e}")
        exit(1)
    
    input_shape = X.shape[1:]  # Input shape for the CNN
    
    # Step 3: Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 4: Build the CNN model
    model = build_model(input_shape, num_classes)
    
    # Step 5: Train the model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Step 6: Save the model
    model.save("cnn_model.h5")
    print("Model saved as 'cnn_model.h5'.")
