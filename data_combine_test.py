import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Read data
data_real = pd.read_excel("data_real.xls")
data_fake = pd.read_excel("data_fake.xls")

# Split input and output
X_fake = data_fake[["V", "R", "D"]]
y_fake = data_fake["I"]

X_real = data_real[["V", "R", "D"]]
y_real = data_real["I"]

# Train xgboost1 using data_fake
xgb1 = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
xgb1.fit(X_fake, y_fake)

# Split data_real into training and testing sets (4:1 ratio)
X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

# Use xgboost1 to predict the last column of the training set
y_train_pred = xgb1.predict(X_train)

# Calculate the residuals (predicted - true values)
residuals_train = y_train_pred - y_train

# Train xgboost2 using the training set (input: first three columns, output: residuals)
xgb2 = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
xgb2.fit(X_train, residuals_train)

# Make predictions on the test set
y_test_pred_xgb1 = xgb1.predict(X_test)  # Predictions from xgboost1
y_test_pred_xgb2 = xgb2.predict(X_test)  # Predictions from xgboost2

# Final predictions are the sum of xgboost1 and xgboost2 outputs
y_test_final_pred = y_test_pred_xgb1 + y_test_pred_xgb2

# Calculate MSE for xgboost1 on the test set
mse_xgb1 = mean_squared_error(y_test, y_test_pred_xgb1)

# Calculate MSE for the final predictions (xgboost1 + xgboost2) on the test set
mse_final = mean_squared_error(y_test, y_test_final_pred)

# Train a new xgboost3 using only the

