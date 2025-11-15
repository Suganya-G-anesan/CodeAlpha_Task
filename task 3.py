import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("car data.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")

df["Owner"] = df["Owner"].astype(str).str.replace(r"[^0-9]", "", regex=True)
df["Owner"] = df["Owner"].replace("", "0").astype(int)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

num_features = ["Year", "Present_Price", "Driven_kms", "Owner"]
cat_features = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]

ct = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
], remainder="passthrough")

model = Pipeline([
    ("preprocess", ct),
    ("rf", RandomForestRegressor(random_state=42))
])

model.fit(X, y)

def predict_price():
    car = input("Car Name: ")
    year = int(input("Year: "))
    present_price = float(input("Present Price (Showroom price in lakhs): "))
    driven = int(input("Driven Kms: "))
    fuel = input("Fuel Type (Petrol / Diesel / CNG): ")
    selling_type = input("Selling type (Dealer / Individual): ")
    transmission = input("Transmission (Manual / Automatic): ")
    owner = int(input("Previous Owners (0/1/2/3): "))

    user_data = pd.DataFrame([{
        "Car_Name": car,
        "Year": year,
        "Present_Price": present_price,
        "Driven_kms": driven,
        "Fuel_Type": fuel,
        "Selling_type": selling_type,
        "Transmission": transmission,
        "Owner": owner
    }])

    price = model.predict(user_data)[0]
    print("\nPredicted Selling Price:", round(price, 2), "Lakhs")

predict_price()
