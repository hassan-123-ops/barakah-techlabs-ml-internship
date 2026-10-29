
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# 1. Create a sample classification dataset
data = {
    "Age": [22, 25, 35, 40, 28, 45, 32, 50, 29, 38,
            24, 42, 31, 55, 27, 48, 36, 44, 26, 39],
    "Salary": [30000, 45000, 55000, 65000, 40000,
               70000, 50000, 80000, 42000, 60000,
               35000, 72000, 48000, 90000, 38000,
               85000, 52000, 68000, 36000, 62000],
    "Experience": [1, 3, 8, 12, 4, 15, 6, 20, 3, 10,
                   2, 14, 5, 25, 2, 18, 7, 13, 1, 11],
    "Purchased": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1,
                  0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df.head())

# 2. Separate features and target
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Create Random Forest model
model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

# 5. Define hyperparameter combinations
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5],
    "min_samples_split": [2, 4],
    "min_samples_leaf": [1, 2]
}

# 6. Perform hyperparameter tuning
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 7. Select the best model
best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(round(grid_search.best_score_, 3))

# 8. Make predictions
y_pred = best_model.predict(X_test)
y_probability = best_model.predict_proba(X_test)[:, 1]

# 9. Evaluate model performance
print("\nTest Accuracy:", round(
    accuracy_score(y_test, y_pred), 3
))

print("Test Precision:", round(
    precision_score(y_test, y_pred, zero_division=0), 3
))

print("Test Recall:", round(
    recall_score(y_test, y_pred, zero_division=0), 3
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

print("Test ROC-AUC:", round(
    roc_auc_score(y_test, y_probability), 3
))

# 10. Save predictions
results = X_test.copy()
results["Actual_Purchased"] = y_test
results["Predicted_Purchased"] = y_pred

results.to_csv("task2_predictions.csv", index=False)

print("\nTask 2 completed successfully!")
print("Predictions saved to task2_predictions.csv")
