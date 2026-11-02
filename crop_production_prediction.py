import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# 2. LOAD DATASnET

df = pd.read_csv(
    r"D:\Project predicyive\Data\cropproduction.csv"
)

print("\n========== DATASET LOADED ==========")
print(df.head())


# 3. BASIC DATASET INFORMATION

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())


# 4. CHECK MISSING VALUES

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# 5. CHECK DUPLICATE ROWS

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())


# 6. DATA CLEANING

# Remove rows where the target variable Production is missing

df_clean = df.dropna(subset=['Production']).copy()

print("\n========== CLEANED DATASET ==========")
print("Original dataset shape:", df.shape)
print("Cleaned dataset shape:", df_clean.shape)

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())


# 7. EXPLORATORY DATA ANALYSIS (EDA)


# 7.1 Production Distribution

plt.figure(figsize=(8, 5))

plt.hist(
    df_clean['Production'],
    bins=50,
    edgecolor='black'
)

plt.title("Distribution of Crop Production")
plt.xlabel("Production")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# 7.2 Top 10 Crops by Total Production

top_crops = (
    df_clean.groupby('Crop')['Production']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))

top_crops.plot(kind='bar')

plt.title("Top 10 Crops by Total Production")
plt.xlabel("Crop")
plt.ylabel("Total Production")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 7.3 Production by Season

season_production = (
    df_clean.groupby('Season')['Production']
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))

season_production.plot(kind='bar')

plt.title("Crop Production by Season")
plt.xlabel("Season")
plt.ylabel("Total Production")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 7.4 Yearly Production Trend

yearly_production = (
    df_clean.groupby('Crop_Year')['Production']
    .sum()
)

plt.figure(figsize=(10, 5))

yearly_production.plot(kind='line', marker='o')

plt.title("Yearly Crop Production Trend")
plt.xlabel("Crop Year")
plt.ylabel("Total Production")
plt.grid(True)
plt.tight_layout()
plt.show()


# 7.5 Area vs Production

plt.figure(figsize=(8, 5))

plt.scatter(
    df_clean['Area'],
    df_clean['Production'],
    alpha=0.5
)

plt.title("Area vs Crop Production")
plt.xlabel("Area")
plt.ylabel("Production")
plt.tight_layout()
plt.show()


# 8. CORRELATION ANALYSIS

correlation_data = df_clean[
    ['Crop_Year', 'Area', 'Production']
]

correlation_matrix = correlation_data.corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation_matrix)

plt.figure(figsize=(7, 5))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# 9. FEATURE SELECTION

X = df_clean[
    [
        'State_Name',
        'District_Name',
        'Crop_Year',
        'Season',
        'Crop',
        'Area'
    ]
]

y = df_clean['Production']

print("\n========== INPUT FEATURES ==========")
print(X.head())

print("\n========== TARGET VARIABLE ==========")
print(y.head())


# 10. ENCODING CATEGORICAL VARIABLES

X_encoded = pd.get_dummies(
    X,
    columns=[
        'State_Name',
        'District_Name',
        'Season',
        'Crop'
    ],
    dtype=int
)

print("\n========== ENCODED DATA ==========")
print(X_encoded.head())

print("Encoded dataset shape:", X_encoded.shape)


# 11. SPLIT DATASET INTO TRAINING AND TESTING DATA

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== TRAINING AND TESTING ==========")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# 12. MODEL 1: LINEAR REGRESSION

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)


# 13. LINEAR REGRESSION EVALUATION

lr_mae = mean_absolute_error(
    y_test,
    lr_predictions
)

lr_mse = mean_squared_error(
    y_test,
    lr_predictions
)

lr_rmse = np.sqrt(lr_mse)

lr_r2 = r2_score(
    y_test,
    lr_predictions
)

print("\n========== LINEAR REGRESSION RESULTS ==========")

print("MAE:", lr_mae)
print("MSE:", lr_mse)
print("RMSE:", lr_rmse)
print("R2 Score:", lr_r2)


# 14. MODEL 2: DECISION TREE REGRESSION

dt_model = DecisionTreeRegressor(
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)


# 15. DECISION TREE EVALUATION

dt_mae = mean_absolute_error(
    y_test,
    dt_predictions
)

dt_mse = mean_squared_error(
    y_test,
    dt_predictions
)

dt_rmse = np.sqrt(dt_mse)

dt_r2 = r2_score(
    y_test,
    dt_predictions
)

print("\n========== DECISION TREE RESULTS ==========")

print("MAE:", dt_mae)
print("MSE:", dt_mse)
print("RMSE:", dt_rmse)
print("R2 Score:", dt_r2)


# 16. MODEL COMPARISON

comparison_results = pd.DataFrame({

    'Model': [
        'Linear Regression',
        'Decision Tree Regression'
    ],

    'MAE': [
        lr_mae,
        dt_mae
    ],

    'MSE': [
        lr_mse,
        dt_mse
    ],

    'RMSE': [
        lr_rmse,
        dt_rmse
    ],

    'R2 Score': [
        lr_r2,
        dt_r2
    ]

})

print("\n========== MODEL COMPARISON ==========")

print(comparison_results)


# 17. VISUALIZE MODEL COMPARISON

plt.figure(figsize=(9, 5))

plt.bar(
    comparison_results['Model'],
    comparison_results['R2 Score']
)

plt.title("Comparison of Model R2 Scores")
plt.xlabel("Machine Learning Models")
plt.ylabel("R2 Score")

plt.tight_layout()
plt.show()


# 18. ACTUAL VS PREDICTED VALUES

results = pd.DataFrame({

    'Actual Production': y_test.values,

    'Linear Regression Prediction': lr_predictions,

    'Decision Tree Prediction': dt_predictions

})

print("\n========== ACTUAL VS PREDICTED ==========")

print(results.head(10))


# 19. ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    lr_predictions,
    alpha=0.5
)

plt.xlabel("Actual Production")
plt.ylabel("Predicted Production")

plt.title("Linear Regression: Actual vs Predicted")

plt.tight_layout()
plt.show()


# 20. PREDICTION ERROR DISTRIBUTION

errors = y_test - lr_predictions

plt.figure(figsize=(8, 5))

plt.hist(
    errors,
    bins=50,
    edgecolor='black'
)

plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# 21. SAVE MODEL COMPARISON RESULTS

comparison_results.to_csv(
    "model_comparison_results.csv",
    index=False
)


# 22. SAVE ACTUAL AND PREDICTED RESULTS

results.to_csv(
    "crop_prediction_results.csv",
    index=False
)


# 23. FINAL PROJECT SUMMARY

print("\n========== FINAL PROJECT SUMMARY ==========")

print("Project: Crop Production Prediction Using Machine Learning")

print("Original Dataset Records:", len(df))

print("Cleaned Dataset Records:", len(df_clean))

print("Total Input Features Before Encoding:", X.shape[1])

print("Total Features After Encoding:", X_encoded.shape[1])

print("\nModel Comparison:")

print(comparison_results)

print("\nProject Execution Completed Successfully!")

# ============================================================
# END OF PROJECT
# ============================================================