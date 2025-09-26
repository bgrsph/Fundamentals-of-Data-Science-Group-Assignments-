from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_africa = pd.read_csv("df_africa.csv")

# Define predictors and target
X = df_africa[["Education_Expenditure_GDP_Pct", "edu_per"]].values  # independent vars
Y = df_africa["Literacy_Rate"].values  # dependent var
years = df_africa["Year"]

# Train-test split

X_train = X[years < 2021]
y_train = Y[years < 2021]
X_test = X[years >= 2021]
y_test = Y[years >= 2021]

# Select model hyperparameters (here, we will use a polynomial basis function, of degree 2)
polymodel = make_pipeline(PolynomialFeatures(2), LinearRegression(fit_intercept=True))

# Fit model to data
polymodel.fit(X_train, y_train)
y_pred = polymodel.predict(X_test)
r2_poly = r2_score(y_test, y_pred)
mse_poly = mean_squared_error(y_test, y_pred)
mae_poly = mean_absolute_error(y_test, y_pred)
print("\nPolynomial Model (Degree 2):")
print(f"R² = {r2_poly:.4f}, MSE = {mse_poly:.4f}, MAE = {mae_poly:.4f}")


# Plot
exp_range = np.linspace(X_train[:,0].min(), X_train[:,0].max(), 100)
edu_fixed = np.full_like(exp_range, X_train[:,1].mean())
X_plot = np.column_stack([exp_range, edu_fixed])
y_plot = polymodel.predict(X_plot)

plt.scatter(X_train[:,0], y_train, color="black", alpha=0.6, label="Data")
plt.plot(exp_range, y_plot, color="red", label="Poly fit (edu_per fixed)")
plt.xlabel("Education Expenditure (% GDP)")
plt.ylabel("Literacy Rate")
plt.title("Literacy Rate vs Education Expenditure")
plt.legend()
plt.show()


# TimeSeriesSplit for cross-validation (good for time series data)
tscv = TimeSeriesSplit(n_splits=5)
# Create grid of parameters to test through cross-validation
param_grid = {'polynomialfeatures__degree': np.arange(20),
              'linearregression__fit_intercept': [True, False]}

model = make_pipeline(PolynomialFeatures(), LinearRegression())
grid = GridSearchCV(model, param_grid, cv=tscv)
grid.fit(X_train, y_train)

# Let us know check the results with the best estimator after Grid Search
print("Best hyperparameters found:")
print(grid.best_params_)
y_pred = grid.best_estimator_.predict(X_test)
print("")

# Compute test error and variance score
print("Model accuracy:")
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print("")


params = {
    "min_samples_split": 5,
    "loss": "squared_error",
    "random_state": 42,
}

# Create a grid of hyperparameters to test through cross-validation
param_grid = {
    'gradientboostingregressor__n_estimators': [500, 1000, 1500],
    'gradientboostingregressor__max_depth': [2, 3, 5, 7],
    'gradientboostingregressor__learning_rate': [0.01, 0.05, 0.1]
}

model = make_pipeline(GradientBoostingRegressor(**params))
grid = GridSearchCV(model, param_grid, cv=tscv)
grid.fit(X_train, y_train)

# Let us know check the results with the best estimator after Grid Search
print("Best hyperparameters found:")
print(grid.best_params_)
print("")

y_pred_gr = grid.best_estimator_.predict(X_test)

results = dict()

# Evaluate the predictions
mse = mean_squared_error(y_test, y_pred_gr)
mae = mean_absolute_error(y_test, y_pred_gr)
r2 = r2_score(y_test, y_pred_gr)
results[f'Fold total'] = {'MSE': mse, 'MAE': mae, 'R2': r2}
print(f"Gradient boosting: MSE: {mse}, MAE: {mae}, R2: {r2}")
# Compute test error and variance score
print("Model accuracy:")
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred_gr))
print('Variance score: %.2f' % r2_score(y_test, y_pred_gr))
print("")

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_gr, alpha=0.7, color="teal")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", lw=2)  # perfect prediction line
plt.xlabel("Actual Literacy Rate (%)")
plt.ylabel("Predicted Literacy Rate (%)")
plt.title("Predicted vs Actual Literacy Rates (Test Set)")
plt.grid(True)
plt.show()

residuals = y_test - y_pred_gr

plt.figure(figsize=(8,6))
plt.scatter(y_pred_gr, residuals, alpha=0.7, color="purple")
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Predicted Literacy Rate (%)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

# Extract feature importances from the best gradient boosting model
feature_importances = grid.best_estimator_.named_steps['gradientboostingregressor'].feature_importances_
features = ["Education Expenditure (% GDP)", "Education Per Capita"]

# Create bar plot
plt.figure(figsize=(8, 6))
plt.bar(features, feature_importances, color=['#4BC0C0', '#9966FF'], edgecolor='black')
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Gradient Boosting Model")
plt.ylim(0, 1)
plt.show()

# Get years for test set
test_years = years[years >= 2021].values

# Create line plot
plt.figure(figsize=(10, 6))
plt.plot(test_years, y_test, 'k-o', label="Actual Literacy Rate", markersize=5)
plt.plot(test_years, y_pred, 'r--o', label="Polynomial Model Predictions", markersize=5)
plt.plot(test_years, y_pred_gr, 'b--o', label="Gradient Boosting Predictions", markersize=5)
plt.xlabel("Year")
plt.ylabel("Literacy Rate")
plt.title("Actual vs Predicted Literacy Rates Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Create grid for contour plot
exp_min, exp_max = X_test[:, 0].min(), X_test[:, 0].max()
edu_min, edu_max = X_test[:, 1].min(), X_test[:, 1].max()
exp_range = np.linspace(exp_min, exp_max, 100)
edu_range = np.linspace(edu_min, edu_max, 100)
exp_grid, edu_grid = np.meshgrid(exp_range, edu_range)
X_grid = np.column_stack([exp_grid.ravel(), edu_grid.ravel()])

# Predict literacy rates for the grid using polynomial model
y_grid = polymodel.predict(X_grid).reshape(exp_grid.shape)

# Create contour plot
plt.figure(figsize=(10, 8))
contour = plt.contourf(exp_grid, edu_grid, y_grid, levels=20, cmap='viridis')
plt.colorbar(contour, label='Predicted Literacy Rate')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', label='Test Data', cmap='viridis')
plt.xlabel("Education Expenditure (% GDP)")
plt.ylabel("Education Per Capita")
plt.title("Polynomial Model Prediction Surface with Test Data")
plt.legend()
plt.show()

# Define KNN model with pipeline
knn_model = make_pipeline(StandardScaler(), KNeighborsRegressor())

# Hyperparameter tuning
param_grid = {
    'kneighborsregressor__n_neighbors': [3, 5, 7, 10, 15],
    'kneighborsregressor__weights': ['uniform', 'distance']
}
tscv = TimeSeriesSplit(n_splits=5)
grid_knn = GridSearchCV(knn_model, param_grid, cv=tscv, scoring='r2')
grid_knn.fit(X_train, y_train)
knn_model = grid_knn.best_estimator_
print("Best KNN parameters:", grid_knn.best_params_)

# Evaluate model
y_pred_train = knn_model.predict(X_train)
y_pred_test = knn_model.predict(X_test)
print("KNN Model Performance:")
print(f"Train R²: {knn_model.score(X_train, y_train):.4f}")
print(f"Test R²: {knn_model.score(X_test, y_test):.4f}")
print(f"Train MSE: {mean_squared_error(y_train, y_pred_train):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, y_pred_test):.4f}")
print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")

# Prediction grid for visualization
exp_range = np.linspace(X_train[:,0].min(), X_train[:,0].max(), 50)  # expenditure
edu_range = np.linspace(X_train[:,1].min(), X_train[:,1].max(), 50)  # education per capita
exp_grid, edu_grid = np.meshgrid(exp_range, edu_range)
X_grid = np.column_stack([exp_grid.ravel(), edu_grid.ravel()])
y_pred_knn = knn_model.predict(X_grid).reshape(exp_grid.shape)

# 3D Plot with train/test points
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(exp_grid, edu_grid, y_pred_knn, cmap="coolwarm", alpha=0.6)
ax.scatter(X_train[:,0], X_train[:,1], y_train, color="black", label="Train", alpha=0.7)
ax.scatter(X_test[:,0], X_test[:,1], y_test, color="red", marker="^", s=60, label="Test")
ax.set_xlabel("Education Expenditure (% GDP)", labelpad=15)
ax.set_ylabel("Education Per Capita", labelpad=15)
ax.set_zlabel("Literacy Rate", labelpad=20)
ax.set_title("KNN Regression Prediction Surface (Train < 2021, Test ≥ 2021)")
ax.view_init(elev=20, azim=45)
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()