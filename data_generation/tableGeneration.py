import os
import random
import pandas as pd
from faker import Faker
from datetime import timedelta

OUTPUT_DIR = "relational_tables"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

fake = Faker()

NUM_DATASETS = 15000

# Global counter for instances where a dummy category column was added.
new_category_count = 0

# Helper Function to Modify Schema

def modify_schema(df, drop_prob=0.2, rename_prob=0.2, shuffle_prob=0.2):
    """
    Modifies the schema of a DataFrame to simulate real-world inconsistencies:
      - With probability drop_prob, drop one random column
      - With probability rename_prob, rename one random column
      - With probability shuffle_prob, shuffle the column order
    """
    # Drop a random column if possible
    if random.random() < drop_prob and len(df.columns) > 1:
        col_to_drop = random.choice(df.columns.tolist())
        df = df.drop(columns=[col_to_drop])

    # Rename a random column
    if random.random() < rename_prob:
        col_to_rename = random.choice(df.columns.tolist())
        new_name = col_to_rename.replace("_", " ")
        new_name = new_name.title() if random.random() < 0.5 else new_name.upper()
        df = df.rename(columns={col_to_rename: new_name})

    # Shuffle the column order
    if random.random() < shuffle_prob:
        df = df.sample(frac=1, axis=1)

    return df

# Helper to Ensure a Repeating Category Column Exists

def ensure_category_column_exists(df, category_name="category", choices=["A", "B", "C"]):
    """
    Checks if any non-numeric column in df has duplicate values.
    If none do, adds a new column (by default named "category") with values drawn from
    a small fixed set. When a new column is added, the global counter new_category_count
    is incremented and an attribute 'dummy_added' is set on the DataFrame.
    """
    non_numeric_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in non_numeric_cols:
        if df[col].duplicated().any():
            return df  # At least one column already repeats.
    global new_category_count
    new_category_count += 1
    df[category_name] = [random.choice(choices) for _ in range(len(df))]
    df.attrs["dummy_added"] = True
    return df

# Helper to Check Validity of a Table

def valid_table(df, min_repeats=10):
    """
    Returns True if at least one non-numeric column in df contains some value that occurs
    at least min_repeats times and no dummy column was added
    """
    if df.attrs.get("dummy_added", False):
        return False
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].value_counts().max() >= min_repeats:
            return True
    return False

# Generate tables in different domains

def generate_customers():
    """
    Modified Customers dataset.
    Instead of using fully random city names and countries, we use a fixed list for cities
    and set the country to "USA" so that these columns naturally have many repeated values.
    """
    row_count = random.randint(110, 150)
    fixed_cities = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"
    ]
    data = {
        "customer_id": list(range(1, row_count + 1)),
        "name": [fake.name() for _ in range(row_count)],
        "email": [fake.email() for _ in range(row_count)],
        "phone": [fake.phone_number() for _ in range(row_count)],
        "address": [fake.street_address() for _ in range(row_count)],
        "city": [random.choice(fixed_cities) for _ in range(row_count)],
        "country": ["USA"] * row_count
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df

def generate_orders():
    row_count = random.randint(110, 200)
    data = {
        "order_id": list(range(1, row_count + 1)),
        "customer_id": [random.randint(1, 100) for _ in range(row_count)],
        "order_date": [fake.date_between(start_date='-1y', end_date='today') for _ in range(row_count)],
        "product": [random.choice(["Widget", "Gadget", "Doodad"]) for _ in range(row_count)],
        "quantity": [random.randint(1, 10) for _ in range(row_count)],
        "price": [round(random.uniform(5, 500), 2) for _ in range(row_count)]
    }
    data["total"] = [data["quantity"][i] * data["price"][i] for i in range(row_count)]
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_products():
    row_count = random.randint(180, 220)
    fixed_categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Toys', 'Books']
    data = {
        "product_id": list(range(1, row_count + 1)),
        "product_name": [fake.word().capitalize() for _ in range(row_count)],
        "category": [random.choice(fixed_categories) for _ in range(row_count)],
        "price": [round(random.uniform(5, 300), 2) for _ in range(row_count)],
        "stock": [random.randint(0, 100) for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    return df


def generate_employees():
    row_count = random.randint(110, 150)
    departments = ['HR', 'Sales', 'IT', 'Finance', 'Marketing']
    data = {
        "employee_id": list(range(1, row_count + 1)),
        "name": [fake.name() for _ in range(row_count)],
        "email": [fake.email() for _ in range(row_count)],
        "phone": [fake.phone_number() for _ in range(row_count)],
        "department": [random.choice(departments) for _ in range(row_count)],
        "salary": [round(random.uniform(30000, 100000), 2) for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_invoices():
    row_count = random.randint(150, 200)
    statuses = ['Paid', 'Unpaid', 'Overdue']
    data = {
        "invoice_id": list(range(1, row_count + 1)),
        "customer_id": [random.randint(1, 100) for _ in range(row_count)],
        "invoice_date": [fake.date_between(start_date='-1y', end_date='today') for _ in range(row_count)],
        "amount": [round(random.uniform(50, 1000), 2) for _ in range(row_count)]
    }
    data["due_date"] = [fake.date_between(start_date='today', end_date='+30d') for _ in range(row_count)]
    data["status"] = [random.choice(statuses) for _ in range(row_count)]
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_reviews():
    row_count = random.randint(130, 240)
    data = {
        "review_id": list(range(1, row_count + 1)),
        "customer_id": [random.randint(1, 100) for _ in range(row_count)],
        "product_id": [random.randint(1, 120) for _ in range(row_count)],
        "review_text": [fake.sentence(nb_words=10) for _ in range(row_count)],
        "rating": [random.choice(["Excellent", "Good", "Average", "Poor"]) for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_patients():
    row_count = random.randint(150, 200)
    diagnoses = ["Diabetes", "Hypertension", "Asthma", "COVID-19", "Flu", "Allergy", "Migraine", "Arthritis",
                 "Depression"]
    treatments = ["Medication", "Therapy", "Surgery", "Observation", "Rehabilitation"]
    insurance_providers = ["Aetna", "Blue Cross", "UnitedHealthcare", "Cigna", "Kaiser Permanente"]
    admission_dates = [fake.date_between(start_date='-2y', end_date='today') for _ in range(row_count)]
    discharge_dates = [ad + timedelta(days=random.randint(1, 30)) for ad in admission_dates]
    data = {
        "patient_id": list(range(1, row_count + 1)),
        "name": [fake.name() for _ in range(row_count)],
        "age": [random.randint(0, 100) for _ in range(row_count)],
        "gender": [random.choice(["Male", "Female", "Other"]) for _ in range(row_count)],
        "diagnosis": [random.choice(diagnoses) for _ in range(row_count)],
        "treatment": [random.choice(treatments) for _ in range(row_count)],
        "admission_date": admission_dates,
        "discharge_date": discharge_dates,
        "doctor": [fake.name() for _ in range(row_count)],
        "insurance_provider": [random.choice(insurance_providers) for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_transactions():
    row_count = random.randint(110, 200)
    data = {
        "transaction_id": list(range(1, row_count + 1)),
        "account_id": [random.randint(1000, 9999) for _ in range(row_count)],
        "transaction_date": [fake.date_between(start_date='-1y', end_date='today') for _ in range(row_count)],
        "transaction_type": [random.choice(["Credit", "Debit"]) for _ in range(row_count)],
        "amount": [round(random.uniform(10, 10000), 2) for _ in range(row_count)],
        "description": [fake.sentence(nb_words=6) for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_tax_records():
    row_count = random.randint(110, 120)
    incomes = [round(random.uniform(30000, 200000), 2) for _ in range(row_count)]
    data = {
        "record_id": list(range(1, row_count + 1)),
        "taxpayer_id": [random.randint(100000000, 999999999) for _ in range(row_count)],
        "name": [fake.name() for _ in range(row_count)],
        "income": incomes,
        "tax_paid": [round(incomes[i] * random.uniform(0.1, 0.4), 2) for i in range(row_count)],
        "filing_date": [fake.date_between(start_date='-1y', end_date='today') for _ in range(row_count)],
        "status": [random.choice(["Filed", "Pending", "Audited", "Rejected"]) for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_lab_results():
    row_count = random.randint(120, 140)
    test_info = {
        "Glucose": {"unit": "mg/dL", "range": "70-110", "min": 70, "max": 110},
        "Cholesterol": {"unit": "mg/dL", "range": "150-240", "min": 150, "max": 240},
        "Hemoglobin": {"unit": "g/dL", "range": "13-17", "min": 13, "max": 17},
        "Vitamin D": {"unit": "ng/mL", "range": "20-50", "min": 20, "max": 50},
        "Calcium": {"unit": "mg/dL", "range": "8.5-10.5", "min": 8.5, "max": 10.5},
        "Platelet Count": {"unit": "10^3/uL", "range": "150-450", "min": 150, "max": 450}
    }
    test_types = list(test_info.keys())
    chosen_tests = [random.choice(test_types) for _ in range(row_count)]
    results, units, ref_ranges = [], [], []
    for t in chosen_tests:
        info = test_info[t]
        units.append(info["unit"])
        ref_ranges.append(info["range"])
        results.append(round(random.uniform(info["min"], info["max"]), 2))
    data = {
        "test_id": list(range(1, row_count + 1)),
        "sample_id": [random.randint(10000, 99999) for _ in range(row_count)],
        "test_type": chosen_tests,
        "result": results,
        "unit": units,
        "reference_range": ref_ranges,
        "test_date": [fake.date_between(start_date='-1y', end_date='today') for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_students():
    row_count = random.randint(130, 150)
    majors = ["Computer Science", "Business", "Biology", "Engineering", "Psychology",
              "Mathematics", "Economics", "English", "History"]
    data = {
        "student_id": list(range(1, row_count + 1)),
        "name": [fake.name() for _ in range(row_count)],
        "age": [random.randint(18, 30) for _ in range(row_count)],
        "gender": [random.choice(["Male", "Female", "Other"]) for _ in range(row_count)],
        "major": [random.choice(majors) for _ in range(row_count)],
        "enrollment_date": [fake.date_between(start_date='-4y', end_date='today') for _ in range(row_count)],
        "gpa": [round(random.uniform(2.0, 4.0), 2) for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_property_listings():
    row_count = random.randint(110, 150)
    property_types = ["Apartment", "House", "Condo"]
    data = {
        "listing_id": list(range(1, row_count + 1)),
        "address": [fake.street_address() for _ in range(row_count)],
        "city": [fake.city() for _ in range(row_count)],
        "state": [fake.state() for _ in range(row_count)],
        "zip": [fake.zipcode() for _ in range(row_count)],
        "price": [round(random.uniform(50000, 1000000), 2) for _ in range(row_count)],
        "bedrooms": [random.randint(1, 5) for _ in range(row_count)],
        "bathrooms": [random.randint(1, 4) for _ in range(row_count)],
        "square_feet": [random.randint(500, 5000) for _ in range(row_count)],
        "listing_date": [fake.date_between(start_date='-2y', end_date='today') for _ in range(row_count)],
        "property_type": [random.choice(property_types) for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_social_media_posts():
    row_count = random.randint(110, 200)
    post_categories = ["News", "Entertainment", "Personal"]
    data = {
        "post_id": list(range(1, row_count + 1)),
        "user_id": [random.randint(1, 1000) for _ in range(row_count)],
        "post_text": [fake.sentence(nb_words=random.randint(5, 15)) for _ in range(row_count)],
        "post_date": [fake.date_time_between(start_date='-1y', end_date='now') for _ in range(row_count)],
        "likes": [random.randint(0, 5000) for _ in range(row_count)],
        "shares": [random.randint(0, 1000) for _ in range(row_count)],
        "comments_count": [random.randint(0, 500) for _ in range(row_count)],
        "post_category": [random.choice(post_categories) for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_bank_accounts():
    row_count = random.randint(110, 160)
    account_types = ["Savings", "Checking", "Business"]
    branches = [fake.city() for _ in range(10)]
    data = {
        "account_id": list(range(1, row_count + 1)),
        "customer_name": [fake.name() for _ in range(row_count)],
        "account_type": [random.choice(account_types) for _ in range(row_count)],
        "balance": [round(random.uniform(100, 100000), 2) for _ in range(row_count)],
        "opened_date": [fake.date_between(start_date='-5y', end_date='-1y') for _ in range(row_count)],
        "branch": [random.choice(branches) for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_flights():
    row_count = random.randint(110, 150)
    airlines = ["Air Alpha", "Beta Airlines", "Gamma Air", "Delta Flights"]
    cities = [fake.city() for _ in range(20)]
    flight_data = []
    for i in range(1, row_count + 1):
        airline = random.choice(airlines)
        departure = random.choice(cities)
        arrival = random.choice([c for c in cities if c != departure])
        dep_dt = fake.date_time_between(start_date='-1y', end_date='now')
        duration = random.randint(60, 600)
        arr_dt = dep_dt + timedelta(minutes=duration)
        price = round(random.uniform(50, 1500), 2)
        flight_data.append({
            "flight_id": i,
            "airline": airline,
            "departure": departure,
            "arrival": arrival,
            "departure_time": dep_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "arrival_time": arr_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_minutes": duration,
            "price": price
        })
    df = pd.DataFrame(flight_data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_weather_reports():
    row_count = random.randint(110, 200)
    weather_stations = ["Station A", "Station B", "Station C"]
    data = {
        "report_id": list(range(1, row_count + 1)),
        "city": [fake.city() for _ in range(row_count)],
        "temperature": [round(random.uniform(-10, 40), 1) for _ in range(row_count)],
        "humidity": [random.randint(10, 100) for _ in range(row_count)],
        "wind_speed": [round(random.uniform(0, 20), 1) for _ in range(row_count)],
        "report_date": [fake.date_between(start_date='-1y', end_date='today') for _ in range(row_count)],
        "weather_station": [random.choice(weather_stations) for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_medical_appointments():
    row_count = random.randint(110, 160)
    statuses = ["Scheduled", "Completed", "Cancelled"]
    reasons = ["Check-up", "Follow-up", "Emergency", "Consultation", "Routine Exam"]
    data = {
        "appointment_id": list(range(1, row_count + 1)),
        "patient_name": [fake.name() for _ in range(row_count)],
        "doctor_name": [fake.name() for _ in range(row_count)],
        "appointment_date": [fake.date_time_between(start_date='-6m', end_date='now') for _ in range(row_count)],
        "reason": [random.choice(reasons) for _ in range(row_count)],
        "status": [random.choice(statuses) for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_courses():
    row_count = random.randint(110, 140)
    departments = ["Computer Science", "Mathematics", "History", "Biology", "Economics", "Physics"]
    semesters = ["Spring", "Summer", "Fall", "Winter"]
    data = {
        "course_id": list(range(1, row_count + 1)),
        "course_name": [f"Intro to {fake.word().capitalize()}" for _ in range(row_count)],
        "department": [random.choice(departments) for _ in range(row_count)],
        "instructor": [fake.name() for _ in range(row_count)],
        "credits": [random.randint(1, 5) for _ in range(row_count)],
        "semester": [f"{random.choice(semesters)} {random.randint(2018, 2023)}" for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_shipments():
    row_count = random.randint(110, 180)
    statuses = ["In Transit", "Delivered", "Delayed", "Cancelled"]
    data = {
        "shipment_id": list(range(1, row_count + 1)),
        "origin": [fake.city() for _ in range(row_count)],
        "destination": [fake.city() for _ in range(row_count)],
        "shipment_date": [fake.date_between(start_date='-1y', end_date='today') for _ in range(row_count)]
    }
    shipment_dates = pd.to_datetime(data["shipment_date"])
    delivery_dates = [(sd + timedelta(days=random.randint(1, 14))).date().isoformat() for sd in shipment_dates]
    data["delivery_date"] = delivery_dates
    data["status"] = [random.choice(statuses) for _ in range(row_count)]
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_retail_store_inventory():
    row_count = random.randint(110, 160)
    categories = ["Electronics", "Clothing", "Grocery", "Home", "Sports", "Toys"]
    data = {
        "item_id": list(range(1, row_count + 1)),
        "item_name": [f"{fake.word().capitalize()} {fake.word().capitalize()}" for _ in range(row_count)],
        "category": [random.choice(categories) for _ in range(row_count)],
        "quantity": [random.randint(1, 200) for _ in range(row_count)],
        "price": [round(random.uniform(1, 500), 2) for _ in range(row_count)],
        "restock_date": [fake.date_between(start_date='-6m', end_date='today') for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_vehicle_registrations():
    row_count = random.randint(110, 160)
    states = ["California", "Texas", "New York", "Florida", "Illinois"]
    vehicle_types = ["Sedan", "SUV", "Truck", "Coupe", "Van", "Convertible"]
    makes = ["Toyota", "Ford", "Chevrolet", "Honda", "Nissan"]
    models = ["Model X", "Model Y", "Model Z", "Series 1", "Series 2"]
    data = {
        "registration_id": list(range(1, row_count + 1)),
        "state": [random.choice(states) for _ in range(row_count)],
        "vehicle_type": [random.choice(vehicle_types) for _ in range(row_count)],
        "make": [random.choice(makes) for _ in range(row_count)],
        "model": [random.choice(models) for _ in range(row_count)],
        "registration_date": [fake.date_between(start_date='-5y', end_date='today') for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_crime_statistics():
    row_count = random.randint(110, 160)
    crime_types = ["Theft", "Assault", "Burglary", "Robbery", "Vandalism", "Fraud"]
    outcomes = ["Solved", "Unsolved", "Pending", "Arrest Made"]
    data = {
        "crime_id": list(range(1, row_count + 1)),
        "city": [fake.city() for _ in range(row_count)],
        "crime_type": [random.choice(crime_types) for _ in range(row_count)],
        "reported_date": [fake.date_between(start_date='-1y', end_date='today') for _ in range(row_count)],
        "outcome": [random.choice(outcomes) for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_restaurant_reviews():
    row_count = random.randint(110, 160)
    cuisines = ["Italian", "Chinese", "Mexican", "Indian", "Japanese", "American"]
    data = {
        "review_id": list(range(1, row_count + 1)),
        "restaurant_name": [fake.company() for _ in range(row_count)],
        "city": [fake.city() for _ in range(row_count)],
        "cuisine": [random.choice(cuisines) for _ in range(row_count)],
        "review_score": [round(random.uniform(1, 5), 1) for _ in range(row_count)],
        "review_date": [fake.date_between(start_date='-1y', end_date='today') for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df


def generate_book_sales():
    row_count = random.randint(110, 160)
    genres = ["Fiction", "Non-Fiction", "Mystery", "Sci-Fi", "Fantasy", "Biography"]
    data = {
        "sale_id": list(range(1, row_count + 1)),
        "book_title": [fake.sentence(nb_words=3) for _ in range(row_count)],
        "author": [fake.name() for _ in range(row_count)],
        "genre": [random.choice(genres) for _ in range(row_count)],
        "price": [round(random.uniform(5, 50), 2) for _ in range(row_count)],
        "sale_date": [fake.date_between(start_date='-2y', end_date='today') for _ in range(row_count)],
        "copies_sold": [random.randint(1, 100) for _ in range(row_count)]
    }
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df

def generate_stocks_data():
    row_count = random.randint(110, 200)
    tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "FB", "NFLX", "NVDA"]
    dates = [fake.date_between(start_date='-1y', end_date='today') for _ in range(row_count)]
    data = {
        "ticker": [random.choice(tickers) for _ in range(row_count)],
        "date": [d.isoformat() for d in dates],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [random.randint(100000, 10000000) for _ in range(row_count)]
    }
    for _ in range(row_count):
        base_price = round(random.uniform(10, 500), 2)
        open_price = base_price
        high_price = round(open_price + random.uniform(0, 10), 2)
        low_price = round(open_price - random.uniform(0, 10), 2)
        low_price = low_price if low_price > 0 else open_price
        close_price = round(random.uniform(low_price, high_price), 2)
        data["open"].append(open_price)
        data["high"].append(high_price)
        data["low"].append(low_price)
        data["close"].append(close_price)
    data["adj_close"] = [round(c + random.uniform(-0.5, 0.5), 2) for c in data["close"]]
    df = pd.DataFrame(data)
    df = modify_schema(df)
    df = ensure_category_column_exists(df)
    return df

def generate_sports_scores():
    row_count = random.randint(110, 160)
    teams = ["Barcelona", "Liverpool", "Real Madrid", "Manchester United", "Bayern Munich", "AC Milan"]
    leagues = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
    data = {
        "match_id": list(range(1, row_count + 1)),
        "team_home": [random.choice(teams) for _ in range(row_count)],
        "team_away": [],
        "score_home": [random.randint(0, 5) for _ in range(row_count)],
        "score_away": [random.randint(0, 5) for _ in range(row_count)],
        "league": [random.choice(leagues) for _ in range(row_count)],
        "match_date": [fake.date_between(start_date='-1y', end_date='today') for _ in range(row_count)]
    }
    for home in data["team_home"]:
        possible = [team for team in teams if team != home]
        data["team_away"].append(random.choice(possible))
    df = pd.DataFrame(data)
    df = modify_schema(df)
    return df

templates = [
    ("customers", generate_customers),
    ("orders", generate_orders),
    ("products", generate_products),
    ("employees", generate_employees),
    ("invoices", generate_invoices),
    ("reviews", generate_reviews),
    ("patients", generate_patients),
    ("transactions", generate_transactions),
    ("tax_records", generate_tax_records),
    ("lab_results", generate_lab_results),
    ("students", generate_students),
    ("property_listings", generate_property_listings),
    ("social_media_posts", generate_social_media_posts),
    ("bank_accounts", generate_bank_accounts),
    ("flights", generate_flights),
    ("weather_reports", generate_weather_reports),
    ("medical_appointments", generate_medical_appointments),
    ("courses", generate_courses),
    ("shipments", generate_shipments),
    ("retail_store_inventory", generate_retail_store_inventory),
    ("stocks_data", generate_stocks_data),
    ("sports_scores", generate_sports_scores),
    ("vehicle_registrations", generate_vehicle_registrations),
    ("crime_statistics", generate_crime_statistics),
    ("restaurant_reviews", generate_restaurant_reviews),
    ("book_sales", generate_book_sales)
]

valid_count = 0
attempt_count = 0
dropped_count = 0

while valid_count < NUM_DATASETS:
    template_name, generator_func = random.choice(templates)
    df = generator_func()
    attempt_count += 1

    # Check that the table has at least one object column with a value repeated >= 10 times
    if not valid_table(df, min_repeats=10):
        dropped_count += 1
        #print(f"Dropped {template_name} (attempt {attempt_count}) - no column with >= 10 repeating values.")
        continue

    filename = os.path.join(OUTPUT_DIR, f"{template_name}_dataset_{valid_count:05d}.csv")
    df.to_csv(filename, index=False)
    valid_count += 1

    if valid_count % 100 == 0:
        print(f"Generated {valid_count} valid datasets (after {attempt_count} attempts, dropped {dropped_count})")

print("Dataset generation complete!")
# print("Total valid datasets generated:", valid_count)
# print("Total dropped tables:", dropped_count)
# print("Total attempts:", attempt_count)
# print("Total instances where dummy category column was added:", new_category_count)
