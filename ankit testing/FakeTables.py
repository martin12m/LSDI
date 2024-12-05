
# import pandas as pd
# import numpy as np
# import string
# import random
# from faker import Faker
# import os
# import sqlite3

# # Initialize Faker
# fake = Faker()

# # Define predefined columns
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

# # Configuration
# NUM_TABLES = 5000
# MAX_ROWS_PER_TABLE = 50
# MAX_COLUMNS_PER_TABLE = 10
# foreign_key_probability = 0.1  # 10% chance to include a foreign key

# # Function to generate random data
# def generate_random_data(column, dtype, num_rows, all_tables):
#     if dtype == 'int':
#         return np.random.randint(1, 1000, size=num_rows)
#     elif dtype == 'float':
#         return np.random.uniform(1.0, 1000.0, size=num_rows).round(2)
#     elif dtype == 'str':
#         if column == 'email':
#             return [fake.email() for _ in range(num_rows)]
#         elif column == 'name':
#             return [fake.name() for _ in range(num_rows)]
#         elif column == 'address':
#             return [fake.address().replace('\n', ', ') for _ in range(num_rows)]
#         elif column == 'phone':
#             return [fake.phone_number() for _ in range(num_rows)]
#         elif column == 'status':
#             return [random.choice(['active', 'inactive', 'pending']) for _ in range(num_rows)]
#         elif column == 'category':
#             return [random.choice(['A', 'B', 'C', 'D']) for _ in range(num_rows)]
#         elif column == 'description':
#             return [fake.text(max_nb_chars=50) for _ in range(num_rows)]
#         else:
#             return [fake.word() for _ in range(num_rows)]
#     elif dtype == 'datetime':
#         return [fake.date_time_between(start_date='-2y', end_date='now') for _ in range(num_rows)]
#     elif dtype == 'bool':
#         return [random.choice([True, False]) for _ in range(num_rows)]
#     else:
#         return [None] * num_rows  # Default fallback

# # Prepare output directory
# output_dir = 'relational_tables_csv'
# os.makedirs(output_dir, exist_ok=True)

# # Initialize storage
# all_tables = {}

# for table_num in range(1, NUM_TABLES + 1):
#     table_name = f'table_{table_num}'
    
#     # Determine number of columns
#     num_columns = random.randint(1, MAX_COLUMNS_PER_TABLE)
#     columns = random.sample(list(PREDEFINED_COLUMNS.keys()), num_columns)
    
#     # Ensure 'id' is present
#     if 'id' not in columns:
#         columns.insert(0, 'id')
    
#     data = {}
#     num_rows = random.randint(1, MAX_ROWS_PER_TABLE)
    
#     for column in columns:
#         # Decide if this column is a foreign key
#         if column.startswith('fk_') or (random.random() < foreign_key_probability and table_num > 1):
#             # Create a foreign key column
#             referenced_table = random.choice(list(all_tables.keys()))
#             referenced_ids = all_tables[referenced_table]['id'].tolist()
#             if referenced_ids:
#                 data[column] = np.random.choice(referenced_ids, size=num_rows)
#             else:
#                 data[column] = np.random.randint(1, 1000, size=num_rows)
#         else:
#             dtype = PREDEFINED_COLUMNS[column]
#             data[column] = generate_random_data(column, dtype, num_rows, all_tables)
    
#     df = pd.DataFrame(data)
#     all_tables[table_name] = df
    
#     # Save to CSV
#     file_path = os.path.join(output_dir, f"{table_name}.csv")
#     df.to_csv(file_path, index=False)
    
#     if table_num % 500 == 0:
#         print(f"Created and saved {table_num} tables.")

# print("All tables have been created and saved successfully.")


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
    'Group': 'str',  # Added 'Group' to ensure compatibility
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
    elif column_name == 'Group':  # Group column explicitly
        return [random.choice(['Group1', 'Group2', 'Group3']) for _ in range(row_count)]
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
    
    df = pd.DataFrame(table_content)
    tables_dict[table_name] = df
    
    # Save the DataFrame to a CSV file
    csv_file_path = os.path.join(output_directory, f"{table_name}.csv")
    df.to_csv(csv_file_path, index=False)
    
    if table_index % 1000 == 0:
        print(f"Generated {table_index} tables successfully.")

