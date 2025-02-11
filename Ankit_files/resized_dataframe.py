# import pandas as pd
# import numpy as np
# import random
# import os

# # ---- Operators and Inverse Operators ----
# def unstack_dataframe(df):
#     try:
#         non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
#         numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()

#         if len(non_numeric_cols) < 2 or len(numeric_cols) == 0:
#             raise ValueError("Table must contain categorical and numeric columns for unstacking.")
        
#         index_cols = non_numeric_cols[:-1]
#         unstack_col = non_numeric_cols[-1]

#         unstacked_df = df.pivot(index=index_cols, columns=unstack_col, values=numeric_cols[0])
#         unstacked_df = unstacked_df.reset_index()

#         return unstacked_df

#     except Exception as e:
#         print(f"Error during unstacking: {e}")
#         return df

# def transpose_dataframe_inverse(df):
#     try:
#         print("\n🛠 Debug: Transposing DataFrame...")

#         # Ensure first column is categorical (Avoid numeric index issues)
#         if df.dtypes[0] != 'object':
#             df.columns = [f"Column_{i}" for i in range(df.shape[1])]

#         # Transpose and reset index
#         transposed_df = df.set_index(df.columns[0]).T.reset_index()

#         # Print debug info
#         print(f"✅ Transposed DataFrame Shape: {transposed_df.shape}")
#         return transposed_df

#     except Exception as error:
#         print(f"⚠ Error during transpose inverse: {error}")
#         return df

# def apply_random_inverse_operator(df):
#     operators = {
#         "unstack": unstack_dataframe,
#         "transpose": transpose_dataframe_inverse
#     }
#     operator_name = random.choice(list(operators.keys()))
#     transformed_df = operators[operator_name](df)
#     return transformed_df, operator_name

# def resize_dataframe(df, target_rows=101, target_cols=50):
#     try:
#         print("\n🛠 Debug: Resizing DataFrame...")

#         # Get the original shape
#         original_rows, original_cols = df.shape

#         # Create a new DataFrame to hold resized data
#         resized_data = []

#         # Iterate through each column
#         for col in df.columns:
#             # Get the original values
#             values = df[col].values.flatten()

#             # If the original number of rows is less than the target, repeat the values
#             if original_rows < target_rows:
#                 # Repeat the values to fill the target rows
#                 repeated_values = np.tile(values, (target_rows // original_rows) + 1)[:target_rows]
#             else:
#                 # Truncate the values to fit the target rows
#                 repeated_values = values[:target_rows]

#             resized_data.append(repeated_values)

#         # Create resized DataFrame
#         resized_df = pd.DataFrame(np.array(resized_data).T)

#         # Adjust the number of columns
#         if original_cols < target_cols:
#             # If there are fewer columns than target, repeat the columns
#             for i in range(target_cols - original_cols):
#                 resized_df[f'Extra_Feature_{i+1}'] = resized_df.iloc[:, i % original_cols]
#         else:
#             # Truncate the DataFrame to the target number of columns
#             resized_df = resized_df.iloc[:, :target_cols]

#         print(f"✅ Resized DataFrame Shape: {resized_df.shape}")
#         return resized_df

#     except Exception as e:
#         print(f"⚠ Error during resizing: {e}")
#         return df

# def process_files(input_folder, output_folder):
#     # Create output directory if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     # Iterate through all CSV files in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith('.csv'):
#             file_path = os.path.join(input_folder, filename)
#             print(f"Processing file: {file_path}")

#             try:
#                 # Load the CSV file with specified encoding
#                 relational_data = pd.read_csv(file_path, encoding='latin1')  # or use 'ISO-8859-1'

#                 # Apply a random inverse operator to the relational table
#                 non_relational_table, operator_name = apply_random_inverse_operator(relational_data)
#                 print(f"Applied random inverse operator: {operator_name}")

#                 if non_relational_table.empty:
#                     print("Transformed table is empty. Skipping.")
#                     continue

#                 resized_df = resize_dataframe(non_relational_table)
#                 resized_file_path = os.path.join(output_folder, f"resized_{filename}")
#                 resized_df.to_csv(resized_file_path, index=False)
#                 print(f"Saved resized table to: {resized_file_path}")

#             except Exception as e:
#                 print(f"Error processing file {filename}: {e}")

# def main():
#     input_folder = "kaggle_files"
#     output_folder = "resized_kaggle_files"
#     process_files(input_folder, output_folder)

# if __name__ == "__main__":
#     main()





# import os
# import pandas as pd
# import numpy as np
# import random
# from faker import Faker

# # Configuration parameters
# TOTAL_TABLES = 10
# ROW_COUNT = 200  # Fixed number of rows
# COLUMN_COUNT = 100  # Fixed number of columns
# OUTPUT_DIRECTORY = "homogeneous_tables_csv"

# # Initialize Faker for generating realistic data
# faker = Faker()

# # Define column categories for homogeneity
# COLUMN_CATEGORIES = {
#     'Numeric': ['price', 'quantity', 'score', 'rating'],
#     'Categorical': ['category', 'region', 'status', 'group'],
#     'Datetime': ['created_at'],
#     'Boolean': ['active'],
#     'String': ['name', 'product', 'brand']
# }

# # Function to generate mixed homogeneous data
# def generate_data(column_type, row_count):
#     if column_type == 'Numeric':
#         return np.random.uniform(1.0, 1000.0, size=row_count).round(2)
#     elif column_type == 'Categorical':
#         return [random.choice(['A', 'B', 'C', 'D']) for _ in range(row_count)]
#     elif column_type == 'Datetime':
#         return [faker.date_time_between(start_date='-2y', end_date='now') for _ in range(row_count)]
#     elif column_type == 'Boolean':
#         return [random.choice([True, False]) for _ in range(row_count)]
#     elif column_type == 'String':
#         return [faker.word() for _ in range(row_count)]
#     else:
#         return [None] * row_count  # Fallback for unknown types

# # Create output directory if it doesn't exist
# os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# # Generate multiple tables
# for table_index in range(1, TOTAL_TABLES + 1):
#     table_name = f'table_{table_index}'

#     table_content = {}
#     category_list = list(COLUMN_CATEGORIES.keys())

#     # Ensure every table has mixed data types
#     for col_index in range(COLUMN_COUNT):
#         column_type = random.choice(category_list)  # Randomly assign column type
#         column_name = f"{column_type}_{col_index + 1}"
#         table_content[column_name] = generate_data(column_type, ROW_COUNT)

#     # Convert to DataFrame
#     df = pd.DataFrame(table_content)

#     # Save the DataFrame to a CSV file
#     csv_file_path = os.path.join(OUTPUT_DIRECTORY, f"{table_name}.csv")
#     df.to_csv(csv_file_path, index=False)

#     print(f"✅ Generated {table_name} with mixed data types (Size: {df.shape})")

# print(f"\n🎉 All {TOTAL_TABLES} mixed-type homogeneous tables generated successfully and saved in '{OUTPUT_DIRECTORY}'!")


import os
import pandas as pd
import numpy as np
import random
from faker import Faker

# Configuration
TOTAL_TABLES = 15000
ROW_COUNT = 200  # Fixed number of rows
COLUMN_COUNT = 100  # Fixed number of columns
OUTPUT_DIRECTORY = "datasets"
faker = Faker()

# Define categories for homogeneous columns
TABLE_TYPES = ["Finance", "Healthcare", "E-commerce"]

def generate_fake_data(rows=200, cols=100, table_type="E-commerce"):
    """ Generate fake tabular data based on the specified table type """

    if table_type == "Finance":
        table_data = {
            "account_id": [faker.uuid4() for _ in range(rows)],
            "customer_name": [faker.name() for _ in range(rows)],
            "transaction_date": [faker.date_this_decade() for _ in range(rows)],
            "transaction_type": [random.choice(["Deposit", "Withdrawal", "Transfer"]) for _ in range(rows)],
            "amount": [round(random.uniform(100, 5000), 2) for _ in range(rows)],
            "balance": [round(random.uniform(1000, 100000), 2) for _ in range(rows)],
            "currency": [random.choice(["USD", "EUR", "JPY"]) for _ in range(rows)],
            "branch": [random.choice(["New York", "London", "Tokyo"]) for _ in range(rows)],
            "fraud_flag": [random.choice([True, False]) for _ in range(rows)],
            "loan_approval": [random.choice(["Approved", "Rejected", "Pending"]) for _ in range(rows)]
        }

    elif table_type == "Healthcare":
        table_data = {
            "patient_id": [faker.uuid4() for _ in range(rows)],
            "patient_name": [faker.name() for _ in range(rows)],
            "age": [random.randint(20, 90) for _ in range(rows)],
            "gender": [random.choice(["Male", "Female", "Other"]) for _ in range(rows)],
            "diagnosis": [random.choice(["Diabetes", "Hypertension", "Asthma", "None"]) for _ in range(rows)],
            "admission_date": [faker.date_this_decade() for _ in range(rows)],
            "treatment_cost": [round(random.uniform(500, 10000), 2) for _ in range(rows)],
            "insurance_provider": [random.choice(["Aetna", "BlueCross", "Medicare"]) for _ in range(rows)],
            "is_insured": [random.choice([True, False]) for _ in range(rows)]
        }

    elif table_type == "E-commerce":
        table_data = {
            "order_id": [faker.uuid4() for _ in range(rows)],
            "customer_id": [faker.uuid4() for _ in range(rows)],
            "product_name": [faker.word() for _ in range(rows)],
            "category": [random.choice(["Electronics", "Clothing", "Home", "Sports"]) for _ in range(rows)],
            "price": [round(random.uniform(5, 500), 2) for _ in range(rows)],
            "quantity": [random.randint(1, 10) for _ in range(rows)],
            "status": [random.choice(["Pending", "Shipped", "Delivered", "Canceled"]) for _ in range(rows)],
            "payment_method": [random.choice(["Credit Card", "PayPal", "Bitcoin", "Bank Transfer"]) for _ in range(rows)]
        }

    else:
        raise ValueError(f"Unknown table type: {table_type}")

    return pd.DataFrame(table_data)

# Create output directory
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Generate tables
for table_index in range(1, TOTAL_TABLES + 1):
    table_type = random.choice(TABLE_TYPES)
    df = generate_fake_data(rows=200, cols=100, table_type=table_type)
    file_path = os.path.join(OUTPUT_DIRECTORY, f"{table_type.lower()}_{table_index}.csv")
    df.to_csv(file_path, index=False)
    if table_index % 1000 == 0:
        print(f"✅ {table_index} tables generated...")