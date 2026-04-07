import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Hardcoded dataset
data = {
    "SqFt": [1500, 2000, 2500, 1200, 3000, 1800],
    "Bedrooms": [3, 4, 4, 2, 5, 3],
    "Age": [10, 5, 3, 15, 1, 8],
    "Price": [250000, 350000, 450000, 200000, 550000, 320000]
}

df = pd.DataFrame(data)


train = df.iloc[:5]
test = df.iloc[5:]

X_train = train.drop("Price", axis=1)
y_train = train["Price"]

X_test = test.drop("Price", axis=1)
y_test = test["Price"]


lr = LinearRegression()
dt = DecisionTreeRegressor()


lr.fit(X_train, y_train)
dt.fit(X_train, y_train)


lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)


lr_mse = mean_squared_error(y_test, lr_pred)
dt_mse = mean_squared_error(y_test, dt_pred)

# Print results
print("Predictions:")
print(f"Linear Regression Prediction: {lr_pred[0]:.2f}")
print(f"Decision Tree Prediction: {dt_pred[0]:.2f}")

print("\nMSE:")
print(f"Linear Regression MSE: {lr_mse:.2f}")
print(f"Decision Tree MSE: {dt_mse:.2f}")