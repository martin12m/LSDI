import os
import pandas as pd
import numpy as np
import random
from faker import Faker

# Configuration parameters
TOTAL_TABLES = 5000
ROW_COUNT = 100  # Fixed number of rows
COLUMN_COUNT = 50  # Fixed number of columns (including 'id')
FOREIGN_KEY_PROBABILITY = 0.1  # 10% chance to include a foreign key

# Initialize Faker for generating realistic data
data_generator = Faker()

# Define the structure of predefined columns
PREDEFINED_COLUMNS = {
    'id': 'int',
    'name': 'str',
    'age': 'int',
    'email': 'str',
    'address': 'str',
    'phone': 'str',
    'created_at': 'datetime',
    'status': 'str',
    'score': 'float',
    'category': 'str',
    'price': 'float',
    'quantity': 'int',
    'description': 'str',
    'active': 'bool',
    'rating': 'float',
    'label': 'int'  # Add target column
}

# Dynamically add additional columns if needed
for i in range(15, COLUMN_COUNT):
    PREDEFINED_COLUMNS[f'custom_col_{i}'] = random.choice(['int', 'float', 'str'])

# Ensure COLUMN_COUNT does not exceed the total number of available columns
if COLUMN_COUNT > len(PREDEFINED_COLUMNS):
    raise ValueError(f"COLUMN_COUNT ({COLUMN_COUNT}) exceeds available columns ({len(PREDEFINED_COLUMNS)}).")

# Function to create random data based on data type
def generate_data(column_name, data_type, row_count):
    if data_type == 'int':
        return np.random.randint(1, 1000, size=row_count)
    elif data_type == 'float':
        return np.random.uniform(1.0, 1000.0, size=row_count).round(2)
    elif data_type == 'str':
        return generate_string_data(column_name, row_count)
    elif data_type == 'datetime':
        return [data_generator.date_time_between(start_date='-2y', end_date='now') for _ in range(row_count)]
    elif data_type == 'bool':
        return [random.choice([True, False]) for _ in range(row_count)]
    else:
        return [None] * row_count  # Fallback for unknown types

def generate_string_data(column_name, row_count):
    """Generate string data based on the specified column name."""
    if column_name == 'email':
        return [data_generator.email() for _ in range(row_count)]
    elif column_name == 'name':
        return [data_generator.name() for _ in range(row_count)]
    elif column_name == 'address':
        return [data_generator.address().replace('\n', ', ') for _ in range(row_count)]
    elif column_name == 'phone':
        return [data_generator.phone_number() for _ in range(row_count)]
    elif column_name == 'status':
        return [random.choice(['active', 'inactive', 'pending']) for _ in range(row_count)]
    elif column_name == 'category':
        return [random.choice(['A', 'B', 'C', 'D']) for _ in range(row_count)]
    elif column_name == 'description':
        return [data_generator.text(max_nb_chars=50) for _ in range(row_count)]
    else:
        return [data_generator.word() for _ in range(row_count)]

# Create output directory if it doesn't exist
output_directory = 'relational_tables_csv'
os.makedirs(output_directory, exist_ok=True)

# Dictionary to store all generated tables
tables_dict = {}

for table_index in range(1, TOTAL_TABLES + 1):
    table_name = f'table_{table_index}'
    
    # Select exactly COLUMN_COUNT predefined columns
    selected_columns = random.sample(list(PREDEFINED_COLUMNS.keys()), COLUMN_COUNT - 1)
    selected_columns.insert(0, 'id')  # Ensure 'id' is the first column
    
    table_content = {}
    
    for column in selected_columns:
        if random.random() < FOREIGN_KEY_PROBABILITY and table_index > 1:
            # Handle foreign key columns
            foreign_table = random.choice(list(tables_dict.keys()))
            foreign_ids = tables_dict[foreign_table]['id'].tolist()
            if foreign_ids:
                table_content[column] = np.random.choice(foreign_ids, size=ROW_COUNT)
            else:
                table_content[column] = np.random.randint(1, 1000, size=ROW_COUNT)
        else:
            column_type = PREDEFINED_COLUMNS[column]
            table_content[column] = generate_data(column, column_type, ROW_COUNT)
    
    # Add target column 'label' based on the 'score' column
    if 'score' in selected_columns:
        scores = table_content['score']
        table_content['label'] = [1 if score > 500 else 0 for score in scores]  # Example condition for binary classification
    
    df = pd.DataFrame(table_content)
    tables_dict[table_name] = df
    
    # Save the DataFrame to a CSV file
    csv_file_path = os.path.join(output_directory, f"{table_name}.csv")
    df.to_csv(csv_file_path, index=False)
    
    if table_index % 1000 == 0:
        print(f"Generated {table_index} tables successfully.")
        
        
