import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# 1. Sample dataset
data = {
    "Age": [22, 25, np.nan, 35, 28, 40],
    "Salary": [30000, 45000, 50000, np.nan, 40000, 65000],
    "Department": [
        "AI", "IT", "AI", "HR", np.nan, "IT"
    ],
    "Experience": [1, 3, 4, 8, np.nan, 12]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

print("\nMissing Values Before Processing:")
print(df.isnull().sum())


# 2. Custom transformer for Pandas DataFrames
class DataFrameCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.columns:
            X = X.drop(
                columns=self.columns,
                errors="ignore"
            )

        return X


# 3. Define feature columns
numeric_features = ["Age", "Salary", "Experience"]
categorical_features = ["Department"]


# 4. Numerical preprocessing pipeline
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# 5. Categorical preprocessing pipeline
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# 6. Combine both pipelines
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# 7. Complete preprocessing pipeline
pipeline = Pipeline(steps=[
    ("cleaner", DataFrameCleaner()),
    ("preprocessor", preprocessor)
])


# 8. Fit and transform the dataset
processed_data = pipeline.fit_transform(df)


# 9. Get output column names
feature_names = (
    pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

# Convert transformed data into a DataFrame
processed_df = pd.DataFrame(
    processed_data,
    columns=feature_names
)

print("\nProcessed Dataset:")
print(processed_df.round(3))

print("\nMissing Values After Processing:")
print(processed_df.isnull().sum())

# 10. Save processed data
processed_df.to_csv(
    "processed_data.csv",
    index=False
)

print("\nTask 1 completed successfully!")
print("Processed data saved to processed_data.csv")
