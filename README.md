# Linear Regression Analysis

This repository contains an in-depth analysis of a linear regression model. The analysis includes data preprocessing, feature selection, model training, and evaluation using various statistical metrics and visualization techniques.

## Table of Contents
- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
  - [Heatmap](#heatmap)
  - [Scatter Plot](#scatter-plot)
  - [Pairplot](#pairplot)
  - [Dummy Variables](#dummy-variables)
- [Feature Selection](#feature-selection)
  - [Recursive Feature Elimination (RFE)](#recursive-feature-elimination-rfe)
  - [Variance Inflation Factor (VIF)](#variance-inflation-factor-vif)
- [Model Training](#model-training)
  - [Ordinary Least Squares (OLS)](#ordinary-least-squares-ols)
- [Model Evaluation](#model-evaluation)
  - [R² Score and Adjusted R² Score](#r²-score-and-adjusted-r²-score)
  - [Root Mean Squared Error (RMSE)](#root-mean-squared-error-rmse)
  - [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
- [Visualizations](#visualizations)
  - [Predicted vs Actual Values](#predicted-vs-actual-values)
- [Conclusion](#conclusion)

## Introduction
Linear regression is a powerful statistical method used to model the relationship between a dependent variable and one or more independent variables. This repository demonstrates the steps involved in building and evaluating a linear regression model.

## Data Preprocessing

### Heatmap
A heatmap is used to visualize the correlation between different features in the dataset. This helps in identifying multicollinearity and understanding the relationships between variables.

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
```

### Scatter Plot
Scatter plots are used to visualize the relationship between the dependent variable and each independent variable. This helps in understanding the linearity and distribution of data points.

```python
for column in data.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=column, y='dependent_variable', data=data)
    plt.show()
```

### Pairplot
A pairplot is a grid of scatter plots for all pairs of features. It helps in visualizing the relationships between multiple variables at once.

```python
sns.pairplot(data)
plt.show()
```

### Dummy Variables
Dummy variables are created for categorical features to convert them into a numerical format suitable for regression analysis.

```python
data = pd.get_dummies(data, drop_first=True)
```

## Feature Selection

### Recursive Feature Elimination (RFE)
RFE is used to select the most important features by recursively considering smaller sets of features and ranking them by importance.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

model = LinearRegression()
rfe = RFE(model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
selected_features = X.columns[rfe.support_]
```

### Variance Inflation Factor (VIF)
VIF is used to detect multicollinearity between independent variables. Features with high VIF values are considered for removal to improve the model's performance.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)
```

## Model Training

### Ordinary Least Squares (OLS)
OLS is used to train the linear regression model. It estimates the coefficients of the linear equation by minimizing the sum of the squared differences between the observed and predicted values.

```python
import statsmodels.api as sm

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
```

## Model Evaluation

### R² Score and Adjusted R² Score
- **R² Score:** Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Adjusted R² Score:** Adjusts the R² score based on the number of predictors in the model, providing a more accurate measure for multiple regression models.

```python
r2_score = model.rsquared
adjusted_r2_score = model.rsquared_adj
print(f'R² Score: {r2_score}')
print(f'Adjusted R² Score: {adjusted_r2_score}')
```

### Root Mean Squared Error (RMSE)
RMSE measures the average magnitude of the errors between predicted and observed values. It provides an indication of the model's accuracy.

```python
from sklearn.metrics import mean_squared_error
import numpy as np

y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f'RMSE: {rmse}')
```

### Mean Absolute Error (MAE)
MAE measures the average absolute errors between predicted and observed values. It is a simpler metric compared to RMSE but equally important for evaluation.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y, y_pred)
print(f'MAE: {mae}')
```

## Visualizations

### Predicted vs Actual Values
A plot is created to visualize the relationship between predicted and actual values of the dependent variable. This helps in assessing the model's performance and identifying any patterns or discrepancies.

```python
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
```

## Conclusion
This repository provides a comprehensive guide to building and evaluating a linear regression model. By following the steps outlined in this analysis, you can develop a robust linear regression model and assess its performance using various statistical metrics and visualization techniques.
