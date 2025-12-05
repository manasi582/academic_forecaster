# Student Performance Prediction

## 📌 Project Overview
This project is a **Machine Learning application** designed to predict a student's final academic grade (`G3`) and their Pass/Fail status based on various demographic, social, and academic factors. It utilizes the UCI Student Performance Dataset and provides an interactive Streamlit dashboard for real-time predictions.

## ⚙️ Flow of Project
Here is the step-by-step technical breakdown of how the project works:

### 1. Data Acquisition
*   **Script**: `src/download_data.py`
*   **Source**: The system automatically downloads the **Student Performance Dataset** from the UCI Machine Learning Repository.
*   **Process**: It fetches a ZIP file, extracts it, and verifies the presence of the core data files (`student-mat.csv` for Math course data).
*   **Technicality**: Uses `requests` for HTTP retrieval and `zipfile` for extraction.

### 2. Data Preprocessing
*   **Script**: `src/preprocessing.py`
*   **Process**: Before the data can be used by a machine learning model, it is cleaned and formatted.
    *   **Loading**: Reads the semicolon-separated CSV files into a Pandas DataFrame.
    *   **Encoding**: Converts categorical variables (like `sex`, `address`, `Mjob`) into numbers using **One-Hot Encoding** (creating binary columns for each category) or Label Encoding.
    *   **Splitting**: The data is split into a **Training Set (80%)** for teaching the model and a **Test Set (20%)** for evaluating its performance.

### 3. Exploratory Data Analysis (EDA)
*   **Script**: `src/eda.py`
*   **Process**: Analyzes the data to understand patterns before modeling.
    *   **Visualization**: Generates plots to visualize distributions and correlations.
    *   **Key Relationships**: Examines how features like `studytime` and `absences` correlate with the final grade `G3`.
    *   **Outputs**: Saves plots to `results/plots/`.

### 4. Model Development
*   **Script**: `src/model.py`
*   **Process**: Trains two distinct Random Forest models using `scikit-learn`.
    *   **Regression Model (RandomForestRegressor)**:
        *   **Goal**: Predict the exact final grade (0-20).
        *   **Optimization**: Uses `GridSearchCV` to find the best hyperparameters (e.g., `n_estimators`, `max_depth`).
        *   **Performance**: Achieved an **R² score of ~0.81**.
    *   **Classification Model (RandomForestClassifier)**:
        *   **Goal**: Predict if a student passes (Grade >= 10) or fails.
        *   **Performance**: Achieved **~92% accuracy**.
    *   **Persistence**: Saves trained models as `.joblib` files in `models/`.

### 5. Evaluation
*   **Script**: `src/evaluate.py`
*   **Process**: Assesses how well the models perform on unseen data.
    *   **Metrics**: Calculates RMSE for regression and Accuracy/F1-Score for classification.
    *   **Feature Importance**: Identifies which factors matter most (e.g., **Previous Grades (G1, G2)** and **Absences**).
    *   **Visualization**: Generates "Predicted vs Actual" and "Feature Importance" plots.

### 6. Interactive Dashboard
*   **Script**: `src/app.py`
*   **Process**: A user-friendly web interface built with **Streamlit**.
    *   **User Input**: Allows teachers/users to input student details via sliders and dropdowns.
    *   **Real-time Prediction**: Loads the saved models, processes user input, and displays the predicted final grade and Pass/Fail status.

## 🚀 How to Run

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

## 📂 Project Structure
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
