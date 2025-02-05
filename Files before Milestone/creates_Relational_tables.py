# ###CREATE TABLES

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
# NUM_TABLES = 200
# MAX_ROWS_PER_TABLE = 100
# MAX_COLUMNS_PER_TABLE = 4
# foreign_key_probability = 0.0  # 10% chance to include a foreign key

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
# output_dir = 'relational_tables_csv_botafogoTEST'
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




import pandas as pd
import numpy as np
from faker import Faker
from random import sample, randrange, random, choice
from os import makedirs, path

# Initialize Faker instance
fake_gen = Faker()

# Define the schema for the data
COLUMN_TYPES = {
    'id': 'int', 'name': 'str', 'age': 'int', 'email': 'str',
    'address': 'str', 'phone': 'str', 'created_at': 'datetime',
    'status': 'str', 'score': 'float', 'category': 'str',
    'price': 'float', 'quantity': 'int', 'description': 'str',
    'active': 'bool', 'rating': 'float'
}

# Constants for table generation
TOTAL_TABLES, MAX_ROWS, MAX_COLS = 2000, 100, 4
FOREIGN_KEY_CHANCE = 0.0

def generate_column_data(column_name, data_type, num_rows):
    if data_type == 'int':
        return np.random.randint(1, 1000, size=num_rows)
    elif data_type == 'float':
        return np.round(np.random.uniform(1.0, 1000.0, size=num_rows), 2)
    elif data_type == 'str':
        return generate_string_data(column_name, num_rows)
    elif data_type == 'datetime':
        return [fake_gen.date_time_between(start_date='-2y', end_date='now') for _ in range(num_rows)]
    elif data_type == 'bool':
        return [choice([True, False]) for _ in range(num_rows)]
    return [None] * num_rows

def generate_string_data(column_name, num_rows):
    if column_name == 'email':
        return [fake_gen.email() for _ in range(num_rows)]
    elif column_name == 'name':
        return [fake_gen.name() for _ in range(num_rows)]
    elif column_name == 'address':
        return [fake_gen.address().replace('\n', ', ') for _ in range(num_rows)]
    elif column_name == 'phone':
        return [fake_gen.phone_number() for _ in range(num_rows)]
    elif column_name == 'status':
        return [choice(['active', 'inactive', 'pending']) for _ in range(num_rows)]
    elif column_name == 'category':
        return [choice(['A', 'B', 'C', 'D']) for _ in range(num_rows)]
    elif column_name == 'description':
        return [fake_gen.text(max_nb_chars=50) for _ in range(num_rows)]
    return [fake_gen.word() for _ in range(num_rows)]


output_directory = 'relational_tables'
makedirs(output_directory, exist_ok=True)


tables = {}


for table_number in range(1, TOTAL_TABLES + 1):
    table_name = f'table_{table_number}'
    
    num_columns = randrange(1, MAX_COLS + 1)
    selected_columns = sample(list(COLUMN_TYPES.keys()), num_columns)
    
    if 'id' not in selected_columns:
        selected_columns.insert(0, 'id')
    
    data = {}
    num_rows = randrange(1, MAX_ROWS + 1)
    
    for column in selected_columns:
        if column.startswith('fk_') or (random() < FOREIGN_KEY_CHANCE and table_number > 1):
            reference_table = choice(list(tables.keys()))
            reference_ids = tables[reference_table]['id'].tolist()
            data[column] = choice(reference_ids, size=num_rows) if reference_ids else np.random.randint(1, 1000, size=num_rows)
        else:
            column_type = COLUMN_TYPES[column]
            data[column] = generate_column_data(column, column_type, num_rows)
    
    df = pd.DataFrame(data)
    tables[table_name] = df
    
    file_path = path.join(output_directory, f"{table_name}.csv")
    df.to_csv(file_path, index=False)
    
    if table_number % 500 == 0:
        print(f"Generated {table_number} tables.")

print("Relational Tables generation complete.")