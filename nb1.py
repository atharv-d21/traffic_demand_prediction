# importing dependancies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
# loading datasets
training_dataset = pd.read_csv('https://raw.githubusercontent.com/atharv-d21/traffic_demand_prediction/refs/heads/main/datasets/train.csv')
testing_dataset = pd.read_csv('https://raw.githubusercontent.com/atharv-d21/traffic_demand_prediction/refs/heads/main/datasets/test.csv')
training_dataset.head(5)
training_dataset['day'].unique()
training_dataset['RoadType'].unique()
training_dataset['NumberofLanes'].unique()
training_dataset['LargeVehicles'].unique()
training_dataset['Landmarks'].unique()
training_dataset['Temperature'].unique()
training_dataset['Temperature'].max()
training_dataset['Weather'].unique()
training_dataset['demand'].min(), training_dataset['demand'].max()
training_dataset['Index'][training_dataset['Temperature']==training_dataset['Temperature'].min()]
training_dataset['RoadType'].value_counts(dropna=False)
training_dataset['Weather'].value_counts(dropna=False)
# spliting the 'geohast' into multiple features
training_dataset['geohash_prefix'] = training_dataset['geohash'].str[:3]
training_dataset['geohash_location'] = training_dataset['geohash'].str[3:]

testing_dataset['geohash_prefix'] = testing_dataset['geohash'].str[:3]
testing_dataset['geohash_location'] = testing_dataset['geohash'].str[3:]

display(training_dataset.head(5))
training_dataset['geohash_location'].value_counts()
# spliting the 'geohash_location' into multiple feature to be more specific
training_dataset['geohash_sublocation0'] = training_dataset['geohash_location'].str[0]
training_dataset['geohash_sublocation1'] = training_dataset['geohash_location'].str[1]
training_dataset['geohash_sublocation2'] = training_dataset['geohash_location'].str[2]

testing_dataset['geohash_sublocation0'] = testing_dataset['geohash_location'].str[0]
testing_dataset['geohash_sublocation1'] = testing_dataset['geohash_location'].str[1]
testing_dataset['geohash_sublocation2'] = testing_dataset['geohash_location'].str[2]

training_dataset.head(5)
training_dataset['geohash_sublocation0'].value_counts()
training_dataset['geohash_sublocation1'].value_counts()
training_dataset['geohash_sublocation2'].value_counts()
# dropping 'geohash', 'geohash_prefix', 'geohash_location'
training_dataset.drop(['geohash', 'geohash_prefix', 'geohash_location'], axis=1, inplace=True)
testing_dataset.drop(['geohash', 'geohash_prefix', 'geohash_location'], axis=1, inplace=True)

training_dataset.head(5)
training_dataset[['hour', 'minute']] = training_dataset['timestamp'].str.split(':', expand=True)
testing_dataset[['hour', 'minute']] = testing_dataset['timestamp'].str.split(':', expand=True)

# Convert 'hour' and 'minute' to numeric types
training_dataset['hour'] = pd.to_numeric(training_dataset['hour'])
training_dataset['minute'] = pd.to_numeric(training_dataset['minute'])
testing_dataset['hour'] = pd.to_numeric(testing_dataset['hour'])
testing_dataset['minute'] = pd.to_numeric(testing_dataset['minute'])

display(training_dataset.head(5))
# dropping 'timestamp'
training_dataset.drop(['timestamp'], axis=1, inplace=True)
testing_dataset.drop(['timestamp'], axis=1, inplace=True)

training_dataset.head(5)
# --- Part of Day Feature ---
def get_part_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

training_dataset['part_of_day'] = training_dataset['hour'].apply(get_part_of_day)
testing_dataset['part_of_day'] = testing_dataset['hour'].apply(get_part_of_day)

display(training_dataset.head())
# fill null values with 'Unknown'
training_dataset['RoadType'] = training_dataset['RoadType'].fillna("Unknown")
testing_dataset['RoadType'] = testing_dataset['RoadType'].fillna("Unknown")

training_dataset['Weather'] = training_dataset['Weather'].fillna("Unknown")
testing_dataset['Weather'] = testing_dataset['Weather'].fillna("Unknown")
training_dataset['LargeVehicles'] = training_dataset['LargeVehicles'].replace({'Allowed': 1, 'Not Allowed': 0})
testing_dataset['LargeVehicles'] = testing_dataset['LargeVehicles'].replace({'Allowed': 1, 'Not Allowed': 0})

training_dataset['Landmarks'] = training_dataset['Landmarks'].replace({'Yes': 1, 'No': 0})
testing_dataset['Landmarks'] = testing_dataset['Landmarks'].replace({'Yes': 1, 'No': 0})
training_dataset['Temperature'] = training_dataset.groupby('Weather')['Temperature'].transform(lambda x: x.fillna(x.mean()))
testing_dataset['Temperature'] = testing_dataset.groupby('Weather')['Temperature'].transform(lambda x: x.fillna(x.mean()))

display(training_dataset.head(5))
training_dataset['active_hours'] = ((training_dataset['hour'] >= 8) & (training_dataset['hour'] <= 11)) | \
                                   ((training_dataset['hour'] >= 17) & (training_dataset['hour'] <= 20))

testing_dataset['active_hours'] = ((testing_dataset['hour'] >= 8) & (testing_dataset['hour'] <= 11)) | \
                                  ((testing_dataset['hour'] >= 17) & (testing_dataset['hour'] <= 20))

# Convert boolean to integer (1 or 0)
training_dataset['active_hours'] = training_dataset['active_hours'].astype(int)
testing_dataset['active_hours'] = testing_dataset['active_hours'].astype(int)

training_dataset.head(5)
# Identify categorical columns to encode
categorical_cols = ['RoadType', 'Weather', 'part_of_day', 'geohash_sublocation0', 'geohash_sublocation1', 'geohash_sublocation2']

# One-hot encode the training and testing sets
training_dataset = pd.get_dummies(training_dataset, columns=categorical_cols)
testing_dataset = pd.get_dummies(testing_dataset, columns=categorical_cols)

# Ensure both dataframes have the same columns after encoding
# (Aligns test set to train set columns, filling missing with 0)
testing_dataset = testing_dataset.reindex(columns=[col for col in training_dataset.columns if col != 'demand'], fill_value=0)

training_dataset.head()
# Calculate correlations with the 'demand' column
correlations = training_dataset.corr()['demand'].sort_values(ascending=False)

# Display the correlations
print("Correlation of features with demand:")
print(correlations)
# 1. Identify top 10 best and worst features (excluding 'demand' itself)
corr_series = training_dataset.corr()['demand'].drop('demand')
best_10 = corr_series.sort_values(ascending=False).head(10)
worst_10 = corr_series.sort_values(ascending=True).head(10)

# 2. Create 'best_features' (weighted sum of top 10 positive correlations)
training_dataset['best_features'] = training_dataset[best_10.index].mul(best_10).sum(axis=1)
testing_dataset['best_features'] = testing_dataset[best_10.index].mul(best_10).sum(axis=1)

# 3. Create 'worst_features' (weighted sum of top 10 negative correlations)
training_dataset['worst_features'] = training_dataset[worst_10.index].mul(worst_10).sum(axis=1)
testing_dataset['worst_features'] = testing_dataset[worst_10.index].mul(worst_10).sum(axis=1)

print("Top 10 Best Features applied:", list(best_10.index))
print("Top 10 Worst Features applied:", list(worst_10.index))
training_dataset[['best_features', 'worst_features', 'demand']].head()
# Calculate correlations with the 'demand' column
correlations = training_dataset.corr()['demand'].sort_values(ascending=False)

# Display the correlations
print("Correlation of features with demand:")
print(correlations)
training_dataset.head(5)
# identifying the features and target for training and evaluation models
features = training_dataset.drop(['demand', 'Index'], axis=1)
target = training_dataset['demand']

test_features = testing_dataset.drop(['Index'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Initialize the XGBoost Regressor model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)
# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model using Mean Squared Error
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"XGBoost Regressor Mean Squared Error: {mse_xgb}")
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=3,  # 3-fold cross-validation
                           n_jobs=-1, # Use all available cores
                           verbose=1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)
# Print the best parameters and best score
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation MSE: {-grid_search.best_score_}")

# Get the best model
best_xgb_model = grid_search.best_estimator_

# Make predictions on the test set with the best model
y_pred_tuned_xgb = best_xgb_model.predict(X_test)

# Evaluate the tuned model
mse_tuned_xgb = mean_squared_error(y_test, y_pred_tuned_xgb)
print(f"Tuned XGBoost Regressor Mean Squared Error on Test Set: {mse_tuned_xgb}")
import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importances from the best model
feature_importances = best_xgb_model.feature_importances_

# Get feature names from X_train
feature_names = X_train.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the top N important features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title('Top 20 Feature Importances from Tuned XGBoost Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 8))
sns.scatterplot(x=y_test, y=y_pred_tuned_xgb, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Perfect prediction line
plt.title('Actual vs. Predicted Demand (Tuned XGBoost Model)')
plt.xlabel('Actual Demand')
plt.ylabel('Predicted Demand')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
from sklearn.metrics import r2_score

# Calculate R-squared score for the tuned model
r2 = r2_score(y_test, y_pred_tuned_xgb)

print(f"R-squared score for the tuned XGBoost model: {r2}")
test_predictions = best_xgb_model.predict(test_features)

print("Predictions on the testing dataset have been made.")
print(f"First 5 predictions: {test_predictions[:5]}")
submission = pd.DataFrame({
    'Index': testing_dataset['Index'],
    'demand': test_predictions
})

submission.to_csv('submission300502.csv', index=False)

print("Submission file 'submission300502.csv' created successfully.")
print(submission.head())