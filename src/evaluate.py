import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from preprocessing import load_data, preprocess_data, split_data

def evaluate_model(subject='mat'):
    print(f"Evaluating models for {subject}...")
    
    # Load data and models
    df = load_data(subject)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    regressor = joblib.load(f'models/rf_regressor_{subject}.joblib')
    
    # Create results directory
    output_dir = 'results/plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Feature Importance (Regressor)
    importances = regressor.feature_importances_
    feature_names = X.columns
    
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
    plt.title(f'Top 10 Feature Importance (Regressor) - {subject}')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_{subject}.png')
    plt.close()
    print(f"Saved feature importance plot to {output_dir}/feature_importance_{subject}.png")
    
    # 2. Predicted vs Actual (Regressor)
    y_pred = regressor.predict(X_test)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Grade (G3)')
    plt.ylabel('Predicted Grade (G3)')
    plt.title(f'Predicted vs Actual Grades - {subject}')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/predicted_vs_actual_{subject}.png')
    plt.close()
    print(f"Saved predicted vs actual plot to {output_dir}/predicted_vs_actual_{subject}.png")

if __name__ == "__main__":
    evaluate_model('mat')
