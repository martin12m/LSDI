import os
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

def connect_to_database(db_name):
    """Connect to the MySQL database"""
    try:
        connection_string = f"mysql+mysqlconnector://guest:relational@db.relational-data.org:3306/{db_name}"
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        print(f"Error connecting to MySQL database {db_name}: {e}")
        return None


def get_database_names(conn):
    """Retrieve the names of all databases"""
    try:
        query = "SHOW DATABASES"
        databases = pd.read_sql(query, conn)
        return [db[0] for db in databases.values if db[0] not in ('information_schema', 'mysql', 'performance_schema', 'sys', 'tmp')]
    except Exception as e:
        print(f"Error fetching database names: {e}")
        return []


def get_table_names(conn):
    """Retrieve the names of all tables in the connected database"""
    try:
        query = "SHOW TABLES"
        tables = pd.read_sql(query, conn)
        return [table[0] for table in tables.values]
    except Exception as e:
        print(f"Error fetching table names: {e}")
        return []


def fetch_table_data(conn, table_name):
    """Fetch all data from a specific table"""
    try:
        # Escape table name with backticks to handle reserved keywords
        query = f"SELECT * FROM `{table_name}`"
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"Error fetching data from table {table_name}: {e}")
        return None


def save_to_csv(df, output_path):
    if os.path.exists(output_path):
        #print(f"File {output_path} already exists, skipping.")
        return
    try:
        df.to_csv(output_path, index=False)
        #print(f"Saved table to CSV: {output_path}")
    except Exception as e:
        print(f"Error saving table to CSV: {e}")


def process_all_databases():
    """Iterate through all databases, connect to them, fetch tables, and save them as .csv files"""
    # folder to save the .csv files
    csv_intermediate_folder = "datasets_csv"
    os.makedirs(csv_intermediate_folder, exist_ok=True)

    # connect to the MySQL server
    conn = connect_to_database('information_schema')
    if not conn:
        return

    # get all database names
    database_names = get_database_names(conn)
    if not database_names:
        print("No databases found.")
        conn.dispose()
        return

    for db_name in database_names:
        #print(f"Connecting to database: {db_name}")
        conn = connect_to_database(db_name)
        if not conn:
            continue

        table_names = get_table_names(conn)
        if not table_names:
            print(f"No tables found in the database: {db_name}")
            conn.dispose()
            continue

        with tqdm(total=len(table_names), desc=f"Processing {db_name}") as progress:
            for table_name in table_names:
                df = fetch_table_data(conn, table_name)
                if df is not None:
                    output_path = os.path.join(csv_intermediate_folder, f"{db_name}_{table_name}.csv")
                    save_to_csv(df, output_path)
                progress.update(1)

        # fetch the data and save table to .csv
        for table_name in table_names:
            df = fetch_table_data(conn, table_name)
            if df is not None:
                output_path = os.path.join(csv_intermediate_folder, f"{db_name}_{table_name}.csv")
                save_to_csv(df, output_path)

        conn.dispose()