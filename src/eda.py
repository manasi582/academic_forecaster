import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from preprocessing import load_data

def run_eda(subject='mat'):
    print(f"Running EDA for {subject}...")
    df = load_data(subject)
    
    # Create results directory
    output_dir = 'results/plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Distribution of Final Grade (G3)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['G3'], kde=True, bins=20)
    plt.title(f'Distribution of Final Grades (G3) - {subject}')
    plt.xlabel('Final Grade (G3)')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/g3_distribution_{subject}.png')
    plt.close()
    print(f"Saved G3 distribution plot to {output_dir}/g3_distribution_{subject}.png")
    
    # 2. Correlation Heatmap (Numerical features)
    plt.figure(figsize=(12, 10))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title(f'Correlation Heatmap - {subject}')
    plt.savefig(f'{output_dir}/correlation_heatmap_{subject}.png')
    plt.close()
    print(f"Saved correlation heatmap to {output_dir}/correlation_heatmap_{subject}.png")
    
    # 3. G3 vs Study Time
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='studytime', y='G3', data=df)
    plt.title(f'Final Grade (G3) vs Study Time - {subject}')
    plt.xlabel('Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)')
    plt.ylabel('Final Grade (G3)')
    plt.savefig(f'{output_dir}/g3_vs_studytime_{subject}.png')
    plt.close()
    print(f"Saved G3 vs Study Time plot to {output_dir}/g3_vs_studytime_{subject}.png")
    
    # 4. G3 vs Absences
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='absences', y='G3', data=df)
    plt.title(f'Final Grade (G3) vs Absences - {subject}')
    plt.xlabel('Number of Absences')
    plt.ylabel('Final Grade (G3)')
    plt.savefig(f'{output_dir}/g3_vs_absences_{subject}.png')
    plt.close()
    print(f"Saved G3 vs Absences plot to {output_dir}/g3_vs_absences_{subject}.png")

if __name__ == "__main__":
    run_eda('mat')
