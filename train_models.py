# Required Imports
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_excel('Data.xlsx')
X = data[['REACTIONS', 'COMMENTS', 'REPOSTS']]
y = data['IMPRESSIONS']




def train_xgb(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=20,
        max_depth=3, 
        learning_rate=0.3,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

def train_poly(X, y):
    # Sets max exponent to 2
    poly = PolynomialFeatures(degree=2)

    # scales the data
    X_poly = poly.fit_transform(X)

    # gets our test split of data rather than all of it 
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")


train_poly(X, y)