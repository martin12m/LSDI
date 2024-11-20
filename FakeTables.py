# import pymysql

# # Database connection setup
# connection = pymysql.connect(
#     host='db.relational-data.org',  # Hostname provided
#     port=3306,                      # Default MySQL port
#     user='guest',                   # Username provided
#     password='relational',          # Password provided
#     database='your_database'        # Replace with your actual database name
# )

# cursor = connection.cursor()

# # Create 5000 tables
# for i in range(1, 5001):
#     table_name = f"table_{i}"  # Table names will be table_1, table_2, ..., table_5000
#     create_table_query = f"""
#     CREATE TABLE {table_name} (
#         id INT AUTO_INCREMENT PRIMARY KEY,
#         name VARCHAR(255),
#         value INT,
#         created_at DATETIME
#     );
#     """
#     cursor.execute(create_table_query)
#     print(f"Created table: {table_name}")

# # Commit changes and close the connection
# connection.commit()
# cursor.close()
# connection.close()

# print("All tables have been created successfully!")


import pandas as pd
import numpy as np
import string
import random
from faker import Faker
import os
import sqlite3

# Initialize Faker
fake = Faker()

# Define predefined columns
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
    'rating': 'float'
}

# Configuration
NUM_TABLES = 5000
MAX_ROWS_PER_TABLE = 50
MAX_COLUMNS_PER_TABLE = 10
foreign_key_probability = 0.1  # 10% chance to include a foreign key

# Function to generate random data
def generate_random_data(column, dtype, num_rows, all_tables):
    if dtype == 'int':
        return np.random.randint(1, 1000, size=num_rows)
    elif dtype == 'float':
        return np.random.uniform(1.0, 1000.0, size=num_rows).round(2)
    elif dtype == 'str':
        if column == 'email':
            return [fake.email() for _ in range(num_rows)]
        elif column == 'name':
            return [fake.name() for _ in range(num_rows)]
        elif column == 'address':
            return [fake.address().replace('\n', ', ') for _ in range(num_rows)]
        elif column == 'phone':
            return [fake.phone_number() for _ in range(num_rows)]
        elif column == 'status':
            return [random.choice(['active', 'inactive', 'pending']) for _ in range(num_rows)]
        elif column == 'category':
            return [random.choice(['A', 'B', 'C', 'D']) for _ in range(num_rows)]
        elif column == 'description':
            return [fake.text(max_nb_chars=50) for _ in range(num_rows)]
        else:
            return [fake.word() for _ in range(num_rows)]
    elif dtype == 'datetime':
        return [fake.date_time_between(start_date='-2y', end_date='now') for _ in range(num_rows)]
    elif dtype == 'bool':
        return [random.choice([True, False]) for _ in range(num_rows)]
    else:
        return [None] * num_rows  # Default fallback

# Prepare output directory
output_dir = 'relational_tables_csv'
os.makedirs(output_dir, exist_ok=True)

# Initialize storage
all_tables = {}

for table_num in range(1, NUM_TABLES + 1):
    table_name = f'table_{table_num}'
    
    # Determine number of columns
    num_columns = random.randint(1, MAX_COLUMNS_PER_TABLE)
    columns = random.sample(list(PREDEFINED_COLUMNS.keys()), num_columns)
    
    # Ensure 'id' is present
    if 'id' not in columns:
        columns.insert(0, 'id')
    
    data = {}
    num_rows = random.randint(1, MAX_ROWS_PER_TABLE)
    
    for column in columns:
        # Decide if this column is a foreign key
        if column.startswith('fk_') or (random.random() < foreign_key_probability and table_num > 1):
            # Create a foreign key column
            referenced_table = random.choice(list(all_tables.keys()))
            referenced_ids = all_tables[referenced_table]['id'].tolist()
            if referenced_ids:
                data[column] = np.random.choice(referenced_ids, size=num_rows)
            else:
                data[column] = np.random.randint(1, 1000, size=num_rows)
        else:
            dtype = PREDEFINED_COLUMNS[column]
            data[column] = generate_random_data(column, dtype, num_rows, all_tables)
    
    df = pd.DataFrame(data)
    all_tables[table_name] = df
    
    # Save to CSV
    file_path = os.path.join(output_dir, f"{table_name}.csv")
    df.to_csv(file_path, index=False)
    
    if table_num % 500 == 0:
        print(f"Created and saved {table_num} tables.")

print("All tables have been created and saved successfully.")