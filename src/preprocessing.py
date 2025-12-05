import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(subject='mat'):
    """
    Load student performance data.
    subject: 'mat' (Math) or 'por' (Portuguese)
    """
    if subject == 'mat':
        file_path = 'data/student-mat.csv'
    elif subject == 'por':
        file_path = 'data/student-por.csv'
    else:
        raise ValueError("Subject must be 'mat' or 'por'")
    
    # The CSVs are semicolon separated
    df = pd.read_csv(file_path, sep=';')
    return df

def preprocess_data(df):
    """
    Preprocess the dataframe:
    - Encode categorical variables
    - Split into features (X) and target (y)
    """
    # Copy df to avoid modifying original
    data = df.copy()
    
    # Identify categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Use Label Encoding for binary variables and One-Hot for others?
    # Actually, for Random Forest, Label Encoding is often fine, but One-Hot is safer for interpretation.
    # Let's use pd.get_dummies for simplicity and effectiveness.
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Separate features and target
    # Target is G3
    X = data.drop('G3', axis=1)
    y = data['G3']
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    # Test the functions
    print("Loading Math data...")
    df = load_data('mat')
    print(f"Shape: {df.shape}")
    print(df.head())
    
    print("\nPreprocessing...")
    X, y = preprocess_data(df)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Test shape: {X_test.shape}, {y_test.shape}")
    
    print("\nPreprocessing successful!")
