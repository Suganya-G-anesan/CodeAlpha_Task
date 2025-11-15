import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Advertising.csv")

df.columns = df.columns.str.strip().str.replace('"','')

X = df[["TV","Radio","Newspaper"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

def predict_sales():
    tv = float(input("TV Advertising Spend: "))
    radio = float(input("Radio Advertising Spend: "))
    news = float(input("Newspaper Advertising Spend: "))

    user_data = pd.DataFrame([{
        "TV": tv,
        "Radio": radio,
        "Newspaper": news
    }])

    prediction = model.predict(user_data)[0]

    print("\nPredicted Sales:", round(prediction, 2), "units")

predict_sales()
