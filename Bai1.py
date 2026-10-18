import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('Social_Network_Ads.csv')

df_clean = df.drop(columns=['User ID'])

df_clean['Gender'] = df_clean['Gender'].map({'Male': 1, 'Female': 0})

y = df_clean['Purchased']

experiments = {
    'Full Features (Age, Salary, Gender)': ['Age', 'EstimatedSalary', 'Gender'],
    'Drop Gender (Age, Salary)': ['Age', 'EstimatedSalary'],
    'Drop Salary (Age, Gender)': ['Age', 'Gender'],
    'Drop Age (Salary, Gender)': ['EstimatedSalary', 'Gender']
}

all_records = []

for exp_name, cols in experiments.items():
    X = df_clean[cols]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Linear Regression (Thresh >= 0.5)': LinearRegression(),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
    }
    

    for model_name, model in models.items():
        if 'Linear Regression' in model_name:
            model.fit(X_train_scaled, y_train)
            raw_preds = model.predict(X_test_scaled)
            y_pred = (raw_preds >= 0.5).astype(int)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
        all_records.append({
            'Kịch bản đặc trưng': exp_name,
            'Mô hình': model_name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'F1-Score': round(f1_score(y_test, y_pred, zero_division=0), 4)
        })

results_df = pd.DataFrame(all_records)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(results_df.to_string(index=False))

results_df.to_csv('model_comparison_all_experiments.csv', index=False)
