import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
from preprocessing import load_data, preprocess_data, split_data

def train_models(subject='mat'):
    print(f"Training models for {subject}...")
    df = load_data(subject)
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # --- Regression Model (Predict G3) ---
    print("\nTraining Random Forest Regressor...")
    rf_reg = RandomForestRegressor(random_state=42)
    
    # Simple Grid Search
    param_grid_reg = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    grid_reg = GridSearchCV(rf_reg, param_grid_reg, cv=5, n_jobs=-1, scoring='r2')
    grid_reg.fit(X_train, y_train)
    
    best_reg = grid_reg.best_estimator_
    print(f"Best Regressor Params: {grid_reg.best_params_}")
    
    # Evaluate Regressor
    y_pred_reg = best_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_reg)
    
    print(f"Regressor RMSE: {rmse:.4f}")
    print(f"Regressor R2: {r2:.4f}")
    
    # Save Regressor
    joblib.dump(best_reg, f'models/rf_regressor_{subject}.joblib')
    
    # --- Classification Model (Pass/Fail) ---
    print("\nTraining Random Forest Classifier (Pass/Fail)...")
    # Create binary target: Pass (G3 >= 10) vs Fail (G3 < 10)
    y_class = (y >= 10).astype(int)
    y_train_class = (y_train >= 10).astype(int)
    y_test_class = (y_test >= 10).astype(int)
    
    rf_clf = RandomForestClassifier(random_state=42)
    
    param_grid_clf = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    grid_clf = GridSearchCV(rf_clf, param_grid_clf, cv=5, n_jobs=-1, scoring='accuracy')
    grid_clf.fit(X_train, y_train_class)
    
    best_clf = grid_clf.best_estimator_
    print(f"Best Classifier Params: {grid_clf.best_params_}")
    
    # Evaluate Classifier
    y_pred_clf = best_clf.predict(X_test)
    acc = accuracy_score(y_test_class, y_pred_clf)
    print(f"Classifier Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test_class, y_pred_clf, target_names=['Fail', 'Pass']))
    
    # Save Classifier
    joblib.dump(best_clf, f'models/rf_classifier_{subject}.joblib')
    
    print(f"\nModels saved to models/rf_regressor_{subject}.joblib and models/rf_classifier_{subject}.joblib")

if __name__ == "__main__":
    train_models('mat')
