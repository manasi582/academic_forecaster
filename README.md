# Student Performance Prediction

## Project Overview
This project is a **Machine Learning application** designed to predict a student's final academic grade (`G3`) and their Pass/Fail status based on various demographic, social, and academic factors. It utilizes the UCI Student Performance Dataset and provides an interactive Streamlit dashboard for real-time predictions.

## Flow of Project

### 1. Data Acquisition
*   **Script**: `src/download_data.py`
*   **Details**: Automatically downloads and extracts the Student Performance Dataset from the UCI Machine Learning Repository using `requests` and `zipfile`.

### 2. Data Preprocessing
*   **Script**: `src/preprocessing.py`
*   **Details**: Loads the CSV data, cleans it, and converts categorical variables (e.g., `sex`, `address`) into numerical format using One-Hot Encoding. Splits data into training (80%) and testing (20%) sets.

### 3. Exploratory Data Analysis (EDA)
*   **Script**: `src/eda.py`
*   **Details**: Analyzes data patterns by generating visualizations for grade distributions, feature correlations, and key relationships (e.g., `studytime` vs. `G3`). Saves plots to `results/plots/`.

### 4. Model Development
*   **Script**: `src/model.py`
*   **Details**: Trains two Random Forest models:
    *   **Regressor**: Predicts exact final grade (R² ~0.81).
    *   **Classifier**: Predicts Pass/Fail status (Accuracy ~92%).
    *   Models are optimized using `GridSearchCV` and saved as `.joblib` files.

### 5. Evaluation
*   **Script**: `src/evaluate.py`
*   **Details**: Evaluates model performance using metrics like RMSE and Accuracy. Identifies key features (e.g., previous grades, absences) and generates "Predicted vs Actual" plots.

### 6. Interactive Dashboard
*   **Script**: `src/app.py`
*   **Details**: A Streamlit web app that allows users to input student data and receive real-time grade and pass/fail predictions.

## How to Run

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation
1.  **Clone the repository** (if applicable) or navigate to the project directory.
2.  **Create a virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Execution Steps
Run the following commands in order:

1.  **Download Data**:
    ```bash
    python src/download_data.py
    ```
2.  **Run EDA**:
    ```bash
    python src/eda.py
    ```
3.  **Train Models**:
    ```bash
    python src/model.py
    ```
4.  **Evaluate Models**:
    ```bash
    python src/evaluate.py
    ```
5.  **Launch Dashboard**:
    ```bash
    streamlit run src/app.py
    ```

## Project Structure
```
student_performance/
├── data/                   # Dataset files
├── models/                 # Saved .joblib models
├── results/
│   └── plots/              # Generated EDA and evaluation plots
├── src/
│   ├── app.py              # Streamlit dashboard
│   ├── download_data.py    # Data acquisition script
│   ├── eda.py              # Exploratory Data Analysis script
│   ├── evaluate.py         # Model evaluation script
│   ├── model.py            # Model training script
│   └── preprocessing.py    # Data processing utilities
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```
