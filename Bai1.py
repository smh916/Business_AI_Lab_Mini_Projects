import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('Social_Network_Ads.csv')
print(df.head())
print(df.info())
df = pd.read_csv('Social_Network_Ads.csv')
df_clean = df.drop(columns=['User ID'])
df_clean['Gender'] = df_clean['Gender'].map({'Male': 1, 'Female': 0})
X = df_clean.drop(columns=['Purchased'])
y = df_clean['Purchased']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    results.append({
        'Model': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1-Score': round(f1_score(y_test, y_pred), 4)
    })


lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)


lin_preds = lin_reg.predict(X_test_scaled)
y_pred_lin = (lin_preds >= 0.5).astype(int)

results.append({
    'Model': 'Linear Regression (Thresh >= 0.5)',
    'Accuracy': round(accuracy_score(y_test, y_pred_lin), 4),
    'Precision': round(precision_score(y_test, y_pred_lin), 4),
    'Recall': round(recall_score(y_test, y_pred_lin), 4),
    'F1-Score': round(f1_score(y_test, y_pred_lin), 4)
})


results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
experiments = {
    'Full Features': ['Age', 'EstimatedSalary', 'Gender'],
    'Drop Gender': ['Age', 'EstimatedSalary'],
    'Drop Salary': ['Age', 'Gender'],
    'Drop Age': ['EstimatedSalary', 'Gender']
}

exp_results = []

for exp_name, cols in experiments.items():
    X_exp = df_clean[cols]

    X_tr, X_te, y_tr, y_te = train_test_split(X_exp, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr_sc, y_tr)
    preds = rf.predict(X_te_sc)

    exp_results.append({
        'Kịch bản': exp_name,
        'Số lượng Feature': len(cols),
        'Accuracy': round(accuracy_score(y_te, preds), 4),
        'F1-Score': round(f1_score(y_te, preds), 4)
    })

print(pd.DataFrame(exp_results).to_string(index=False))