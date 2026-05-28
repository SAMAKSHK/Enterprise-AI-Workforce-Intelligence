# ============================================================
# AI WORKFORCE BURNOUT & ATTRITION PREDICTION SYSTEM
# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBClassifier, XGBRegressor

import shap

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("ai_worker_burnout_attrition_2026.csv")

print("\n================ DATASET HEAD ================\n")
print(df.head())

print("\n================ DATASET SHAPE ================\n")
print(df.shape)

print("\n================ DATASET INFO ================\n")
print(df.info())

# ============================================================
# MISSING VALUES
# ============================================================

print("\n================ MISSING VALUES ================\n")
print(df.isnull().sum())

# ============================================================
# BASIC EDA
# ============================================================

# ------------------------------------------------------------
# Burnout Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(
    df['burnout_score'],
    bins=20,
    kde=True
)

plt.title("Burnout Score Distribution")
plt.xlabel("Burnout Score")
plt.ylabel("Count")

plt.show()

# ------------------------------------------------------------
# Attrition Risk Distribution
# ------------------------------------------------------------

plt.figure(figsize=(6,4))

sns.countplot(
    x='attrition_risk',
    data=df
)

plt.title("Attrition Risk Distribution")

plt.show()

# ------------------------------------------------------------
# AI Usage vs Burnout
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

sns.scatterplot(
    x='hours_with_ai_assistance_daily',
    y='burnout_score',
    hue='attrition_risk',
    data=df
)

plt.title("AI Usage vs Burnout")

plt.show()

# ============================================================
# ENCODE CATEGORICAL VARIABLES
# ============================================================

encoded_df = df.copy()

label_encoders = {}

for column in encoded_df.columns:

    if encoded_df[column].dtype == 'object':

        le = LabelEncoder()

        encoded_df[column] = le.fit_transform(encoded_df[column])

        label_encoders[column] = le

print("\n================ ENCODED DATA ================\n")
print(encoded_df.head())

# ============================================================
# ============================================================
# REGRESSION MODEL
# BURNOUT SCORE PREDICTION
# ============================================================
# ============================================================

print("\n================================================")
print("BURNOUT SCORE PREDICTION")
print("================================================\n")

# ============================================================
# FEATURES & TARGET
# ============================================================

X_reg = encoded_df.drop(
    ['burnout_score', 'attrition_risk'],
    axis=1
)

y_reg = encoded_df['burnout_score']

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

# ============================================================
# LINEAR REGRESSION MODEL
# ============================================================

lr_model = LinearRegression()

lr_model.fit(X_train_reg, y_train_reg)

y_pred_lr = lr_model.predict(X_test_reg)

# ============================================================
# LINEAR REGRESSION METRICS
# ============================================================

mae_lr = mean_absolute_error(y_test_reg, y_pred_lr)

rmse_lr = np.sqrt(
    mean_squared_error(y_test_reg, y_pred_lr)
)

r2_lr = r2_score(y_test_reg, y_pred_lr)

print("LINEAR REGRESSION RESULTS\n")

print("MAE :", round(mae_lr,2))
print("RMSE :", round(rmse_lr,2))
print("R2 SCORE :", round(r2_lr,2))

# ============================================================
# OLS STYLE FEATURE IMPACT
# ============================================================

coefficients = pd.DataFrame({

    'Feature': X_reg.columns,
    'Coefficient': lr_model.coef_

})

coefficients = coefficients.sort_values(
    by='Coefficient',
    ascending=False
)

print("\n================ FEATURE IMPACT ================\n")

print(coefficients)

# ============================================================
# XGBOOST REGRESSOR
# ============================================================

xgb_reg = XGBRegressor(

    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42

)

xgb_reg.fit(X_train_reg, y_train_reg)

y_pred_xgb = xgb_reg.predict(X_test_reg)

# ============================================================
# XGBOOST REGRESSION METRICS
# ============================================================

mae_xgb = mean_absolute_error(
    y_test_reg,
    y_pred_xgb
)

rmse_xgb = np.sqrt(
    mean_squared_error(y_test_reg, y_pred_xgb)
)

r2_xgb = r2_score(
    y_test_reg,
    y_pred_xgb
)

print("\nXGBOOST REGRESSION RESULTS\n")

print("MAE :", round(mae_xgb,2))
print("RMSE :", round(rmse_xgb,2))
print("R2 SCORE :", round(r2_xgb,2))

# ============================================================
# ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(8,5))

plt.scatter(
    y_test_reg,
    y_pred_xgb
)

plt.xlabel("Actual Burnout Score")
plt.ylabel("Predicted Burnout Score")

plt.title("Actual vs Predicted Burnout")

plt.show()

# ============================================================
# ============================================================
# CLASSIFICATION MODEL
# ATTRITION RISK PREDICTION
# ============================================================
# ============================================================

print("\n================================================")
print("ATTRITION RISK PREDICTION")
print("================================================\n")

# ============================================================
# FEATURES & TARGET
# ============================================================

X_clf = encoded_df.drop(
    ['attrition_risk', 'burnout_score'],
    axis=1
)

y_clf = encoded_df['attrition_risk']

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(

    X_clf,
    y_clf,
    test_size=0.2,
    random_state=42

)

# ============================================================
# XGBOOST CLASSIFIER
# ============================================================

xgb_clf = XGBClassifier(

    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42

)

xgb_clf.fit(X_train_clf, y_train_clf)

y_pred_clf = xgb_clf.predict(X_test_clf)

# ============================================================
# CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test_clf,
    y_pred_clf
)

precision = precision_score(
    y_test_clf,
    y_pred_clf,
    average='weighted'
)

recall = recall_score(
    y_test_clf,
    y_pred_clf,
    average='weighted'
)

f1 = f1_score(
    y_test_clf,
    y_pred_clf,
    average='weighted'
)

print("CLASSIFICATION RESULTS\n")

print("Accuracy :", round(accuracy,2))
print("Precision :", round(precision,2))
print("Recall :", round(recall,2))
print("F1 Score :", round(f1,2))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test_clf,
    y_pred_clf
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n================ CLASSIFICATION REPORT ================\n")

print(classification_report(
    y_test_clf,
    y_pred_clf
))

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({

    'Feature': X_clf.columns,
    'Importance': xgb_clf.feature_importances_

})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\n================ FEATURE IMPORTANCE ================\n")

print(importance_df)

# ============================================================
# FEATURE IMPORTANCE VISUALIZATION
# ============================================================

plt.figure(figsize=(10,6))

sns.barplot(

    x='Importance',
    y='Feature',
    data=importance_df

)

plt.title("Feature Importance")

plt.show()

# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

explainer = shap.Explainer(xgb_clf)

shap_values = explainer(X_test_clf)

shap.plots.beeswarm(shap_values[:, :, 0])

# ============================================================
# EXPORT FINAL OUTPUTS FOR POWER BI
# ============================================================

final_output = df.copy()

# Burnout Predictions

final_output['predicted_burnout_score'] = xgb_reg.predict(X_reg)

# Attrition Predictions

final_output['predicted_attrition_risk'] = xgb_clf.predict(X_clf)

# CREATE RISK LEVELS

final_output['burnout_risk_level'] = pd.cut(

    final_output['predicted_burnout_score'],

    bins=[0,40,70,100],

    labels=['Low','Medium','High']

)

# SAVE FINAL CSV

final_output.to_csv(
    "final_predictions.csv",
    index=False
)

print("\n================================================")
print("FINAL FILE SAVED SUCCESSFULLY")
print("================================================")

print("\nSaved File : final_predictions.csv")

# ============================================================
# END OF PROJECT
# ============================================================
