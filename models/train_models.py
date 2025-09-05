# Required Imports
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import joblib






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


if __name__ == "__main__":
    data = pd.read_excel('data/Data.xlsx')
    X = data[['REACTIONS', 'COMMENTS', 'REPOSTS']]
    y = data['IMPRESSIONS']

    model = train_xgb(X, y)

    joblib.dump(model, 'models/xgbmodel.pkl')
