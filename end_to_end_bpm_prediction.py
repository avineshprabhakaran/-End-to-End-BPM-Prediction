# %% [code] {"execution":{"iopub.status.busy":"2025-09-25T05:53:33.955014Z","iopub.execute_input":"2025-09-25T05:53:33.955466Z","iopub.status.idle":"2025-09-25T05:53:34.351845Z","shell.execute_reply.started":"2025-09-25T05:53:33.955440Z","shell.execute_reply":"2025-09-25T05:53:34.350563Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# # # 🎵 End-to-End BPM Prediction Pipeline

# %% [markdown]
# ## 📌 Objective
# The goal of this notebook is to predict **BeatsPerMinute (BPM)** for music tracks using audio-derived features.  
# The task is framed as a **regression problem**, and predictions are submitted in Kaggle competition format.
# 

# %% [markdown]
# # ***Setup and Data Loading***

# %% [code] {"execution":{"iopub.status.busy":"2025-09-25T05:53:39.077299Z","iopub.execute_input":"2025-09-25T05:53:39.077787Z","iopub.status.idle":"2025-09-25T05:53:47.396448Z","shell.execute_reply.started":"2025-09-25T05:53:39.077761Z","shell.execute_reply":"2025-09-25T05:53:47.395519Z"}}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from scipy.stats import skew
import lightgbm as lgb
import xgboost as xgb
import catboost as catb
import optuna 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## ***📊 Dataset Overview***

# %% [code] {"execution":{"iopub.status.busy":"2025-09-25T05:53:48.008709Z","iopub.execute_input":"2025-09-25T05:53:48.009057Z","iopub.status.idle":"2025-09-25T05:53:50.447197Z","shell.execute_reply.started":"2025-09-25T05:53:48.009019Z","shell.execute_reply":"2025-09-25T05:53:50.446229Z"}}
# Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

# %% [code]
display(train)
display(test)

# %% [markdown]
# # ****🔍 Exploratory Data Analysis (EDA)****

# %% [markdown]
# - Inspected data structure, shapes, and basic statistics.  
# - Checked missing values and feature distributions.  
# - Visualized the target (`BeatsPerMinute`) to understand skewness and outliers.  
# - Explored feature correlations to identify the most predictive variables.

# %% [code]
print("--First 5 rows --")
display(train.head())
print("\n--First 5 rows --")
test.head()

# %% [markdown]
# # Display basic information about the dataframes

# %% [code]
print("Train data info:")
display(train.info())
print("\nTest data info:")
display(test.info())

# %% [markdown]
# # Check for missing values

# %% [code]

print("\nMissing values in train data:")
print(train.isnull().sum())


# %% [code]
print("\nMissing values in test data:")
print(test.isnull().sum())

# %% [markdown]
# # Visualize the distribution of the target variable

# %% [code]
plt.figure(figsize=(10, 6))
sns.histplot(train['BeatsPerMinute'], kde=True)
plt.title('Distribution of BeatsPerMinute')
plt.xlabel('BeatsPerMinute')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# ### *****Handle Outliers*****

# %% [code] {"scrolled":true}
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize outliers for df
numerical_cols_df = train.select_dtypes(include=['float64', 'int64']).columns
num_plots = len(numerical_cols_df)
num_cols = 3
num_rows = (num_plots + num_cols - 1) // num_cols

plt.figure(figsize=(15, num_rows * 5))
for i, col in enumerate(numerical_cols_df):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.boxplot(data=train, y=col)
    plt.title(f'Outliers in {col} (Train)')
plt.tight_layout()
plt.show()

# Visualize outliers for test_df
numerical_cols_test_df = test.select_dtypes(include=['float64', 'int64']).columns
num_plots = len(numerical_cols_test_df)
num_cols = 3
num_rows = (num_plots + num_cols - 1) // num_cols

plt.figure(figsize=(15, num_rows * 5))
for i, col in enumerate(numerical_cols_test_df):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.boxplot(data=test, y=col)
    plt.title(f'Outliers in {col} (Test)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ***(optional)***

# %% [code]
def outlier_plot(data, exclude_columns, box_color='pink', median_color='brown', whisker_color='purple'):
    columns = data.drop(exclude_columns, axis=1, errors='ignore').columns
    
    n_cols = len(columns)
    n_rows = (n_cols + 1) // 2 
    
    plt.figure(figsize=(15, 5 * n_rows)) 
    
    for i, column in enumerate(columns, 1):
        plt.subplot(n_rows, 2, i)
        plt.boxplot(
            data[column].dropna(), 
            vert=False,
            patch_artist=True,  
            boxprops=dict(facecolor=box_color, color=whisker_color),
            medianprops=dict(color=median_color),
            whiskerprops=dict(color=whisker_color),
            capprops=dict(color=whisker_color),
            flierprops=dict(marker='o', color=whisker_color, markersize=5)
        )
        plt.title(f'{column}')
        plt.xlabel(column)
        plt.grid(False)

    plt.tight_layout()
    plt.show()

# %% [code]
def remove_outliers(data , outlier_columns):
    
    for column in outlier_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        data = data[
            (data[column] >= lower_bound) & (data[column] <= upper_bound)
        ]
    
    return data

# %% [code]
OUTLIER_COLUMNS = ['AudioLoudness' , 'VocalContent' , 'AcousticQuality' ,'RhythmScore' ,'InstrumentalScore' , 'LivePerformanceLikelihood' , 'TrackDurationMs']

train = remove_outliers(train , OUTLIER_COLUMNS)

# %% [code]
outlier_plot(train, ['BeatsPerMinute' , 'id'])

# %% [markdown]
# # ****Visualize Feature Correlations****

# %% [code]
plt.figure(figsize=(12, 10))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Train Data ')
plt.show()

# %% [code]
plt.figure(figsize=(12, 10))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Train Data')
plt.show()

# %% [markdown]
#  # ****⚙️ Feature Engineering****

# %% [markdown]
# ***Create new features that might be helpful for the model.***

# %% [markdown]
# - Applied transformations to handle skewed distributions (e.g., **logTransformer**, scaling).   
# - Created additional engineered features (e.g., squared terms, interaction terms) to capture non-linear relationships.  
# - Ensured consistent preprocessing across train, validation, and test sets.

# %% [markdown]
# Type one 

# %% [code] {"execution":{"iopub.status.busy":"2025-09-25T05:56:37.681532Z","iopub.execute_input":"2025-09-25T05:56:37.681836Z","iopub.status.idle":"2025-09-25T05:56:37.754158Z","shell.execute_reply.started":"2025-09-25T05:56:37.681813Z","shell.execute_reply":"2025-09-25T05:56:37.753123Z"}}
# Create interaction features
train['Vocal_Instrumental_Ratio'] = train['VocalContent'] / (train['InstrumentalScore'] + 1e-6)
test['Vocal_Instrumental_Ratio'] = test['VocalContent'] / (test['InstrumentalScore'] + 1e-6)

train['Energy_AudioLoudness'] = train['Energy'] * train['AudioLoudness']
test['Energy_AudioLoudness'] = test['Energy'] * test['AudioLoudness']

# Create new feature: AudioPerDuration
train['AudioPerDuration'] = train['AudioLoudness'] / train['TrackDurationMs']
test['AudioPerDuration'] = test['AudioLoudness'] / test['TrackDurationMs']
# Create polynomial features (e.g., square of MoodScore)
train['MoodScore_sq'] = train['MoodScore']**2
test['MoodScore_sq'] = test['MoodScore']**2

# Create features based on combinations of other features
train['Rhythm_Energy'] = train['RhythmScore'] * train['Energy']
test['Rhythm_Energy'] = test['RhythmScore'] * test['Energy']

train['Acoustic_Instrumental'] = train['AcousticQuality'] * train['InstrumentalScore']
test['Acoustic_Instrumental'] = test['AcousticQuality'] * test['InstrumentalScore']

# Create new feature: MoodLikelihoodRatio
# Add a small constant to the denominator to avoid division by zero
train['MoodLikelihoodRatio'] = train['MoodScore'] / (train['LivePerformanceLikelihood'] + 1e-6)
test['MoodLikelihoodRatio'] = test['MoodScore'] / (test['LivePerformanceLikelihood'] + 1e-6)

# %% [markdown]
# ## *****Data preprocessing*****

# %% [markdown]
#   # scale numerical features, and encode categorical features if any.

# %% [markdown]
# ***(Optional)***

# %% [code]
from sklearn.preprocessing import PowerTransformer

# Identify numerical columns from the training data (excluding 'id' and 'BeatsPerMinute')
numerical_cols_train = train.select_dtypes(include=np.number).columns.tolist()
numerical_cols_train.remove('id')
if 'BeatsPerMinute' in numerical_cols_train:
    numerical_cols_train.remove('BeatsPerMinute')

# Ensure the test data has the same numerical columns as the training data
numerical_cols_test = [col for col in numerical_cols_train if col in test.columns]

# Apply Yeo-Johnson transformation
pt_yeo_johnson = PowerTransformer(method='yeo-johnson')

# Fit on training data and transform both train and test data using the training columns
train[numerical_cols_train] = pt_yeo_johnson.fit_transform(train[numerical_cols_train])
test[numerical_cols_test] = pt_yeo_johnson.transform(test[numerical_cols_test])


# %% [markdown]
#  # log transformation

# %% [code] {"execution":{"iopub.status.busy":"2025-09-25T05:56:29.518657Z","iopub.execute_input":"2025-09-25T05:56:29.519000Z","iopub.status.idle":"2025-09-25T05:56:29.907786Z","shell.execute_reply.started":"2025-09-25T05:56:29.518975Z","shell.execute_reply":"2025-09-25T05:56:29.906976Z"}}
# Create a list of columns that are safe for log transformation
log_transform_cols = [
    "RhythmScore","AudioLoudness", "VocalContent", "AcousticQuality", "InstrumentalScore",
    "LivePerformanceLikelihood", "MoodScore", "Energy"
]

# Apply a different transformation to AudioLoudness to make it non-negative
# A common method is to shift the data by adding a positive constant.
# Here we will shift by the absolute minimum value + 1 to ensure all values are positive.
min_loudness = train['AudioLoudness'].min()
train['AudioLoudness_transformed'] = np.log1p(train['AudioLoudness'] - min_loudness + 1)
test['AudioLoudness_transformed'] = np.log1p(test['AudioLoudness'] - min_loudness + 1)

# Now apply log transformation to the remaining numerical columns
train_log = train.copy()
test_log = test.copy()

train_log[log_transform_cols] = np.log1p(train_log[log_transform_cols])
test_log[log_transform_cols] = np.log1p(test_log[log_transform_cols])

print("Train dataframe after log transformation:")
display(train_log.head())

print("\nTest dataframe after log transformation:")
display(test_log.head())

# %% [markdown]
# ## *****Feature Scaling*****

# %% [markdown]
# # MinMaxScaler

# %% [code] {"execution":{"iopub.status.busy":"2025-09-25T05:56:56.769850Z","iopub.execute_input":"2025-09-25T05:56:56.770176Z","iopub.status.idle":"2025-09-25T05:56:57.372124Z","shell.execute_reply.started":"2025-09-25T05:56:56.770150Z","shell.execute_reply":"2025-09-25T05:56:57.371244Z"}}
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Select numerical columns, excluding 'id', that are common to both dataframes
numerical_cols = train.select_dtypes(include=['float64', 'int64']).columns.intersection(test.select_dtypes(include=['float64', 'int64']).columns)

# Initialize StandardScaler
scaler =MinMaxScaler()

# Fit and transform on the training data
train_scaled = train.copy()
train_scaled[numerical_cols] = scaler.fit_transform(train_scaled[numerical_cols])

### Transform the test data using the same scaler
test_scaled = test.copy()
test_scaled[numerical_cols] = scaler.transform(test_scaled[numerical_cols])

#print("Scaled train dataframe:")
#display(train.head())

#print("\nScaled test dataframe:")
#display(test.head())

# %% [markdown]
# # StandardScaler

# %% [markdown]
# (Optional)

# %% [code]

from sklearn.preprocessing import StandardScaler

# Select numerical columns, excluding 'id', that are common to both dataframes
numerical_cols = train.select_dtypes(include=['float64', 'int64']).columns.drop('id').intersection(test.select_dtypes(include=['float64', 'int64']).columns.drop('id'))

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform on the training data
train_scaled = train.copy()
train_scaled[numerical_cols] = scaler.fit_transform(train_scaled[numerical_cols])

# Transform the test data using the same scaler
test_scaled = test.copy()
test_scaled[numerical_cols] = scaler.transform(test_scaled[numerical_cols])

#print("Scaled train dataframe:")
#display(train.head())

#print("\nScaled test dataframe:")
#display(test.head())

# %% [code]
X_test = test.drop(['id'], axis=1)

# %% [code] {"execution":{"iopub.status.busy":"2025-09-25T05:57:20.133428Z","iopub.execute_input":"2025-09-25T05:57:20.133711Z","iopub.status.idle":"2025-09-25T05:57:20.198317Z","shell.execute_reply.started":"2025-09-25T05:57:20.133691Z","shell.execute_reply":"2025-09-25T05:57:20.197097Z"}}
X = train.drop(['id', 'BeatsPerMinute'], axis=1)
y = train['BeatsPerMinute']



print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_test shape:", X_test.shape)

# %% [code] {"execution":{"iopub.status.busy":"2025-09-25T05:58:01.633683Z","iopub.execute_input":"2025-09-25T05:58:01.633987Z","iopub.status.idle":"2025-09-25T05:58:01.833607Z","shell.execute_reply.started":"2025-09-25T05:58:01.633956Z","shell.execute_reply":"2025-09-25T05:58:01.832294Z"}}
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)

# %% [markdown]
# # Training and evaluation

# %% [markdown]
# ## 🤖 Modeling
# We trained and evaluated multiple regression models using **RMSE** as the evaluation metric:

# %% [markdown]
# # LGBM model using optuna 

# %% [markdown]
#  **LightGBM with Optuna Tuning**
#    - Performed hyperparameter optimization with Optuna.  
#    - Selected the best parameters and retrained the model on the full dataset.  
#    - Predictions saved as `submissionb.csv`.

# %% [code] {"scrolled":true}
import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 512),
        'max_depth': trial.suggest_int('max_depth', -1, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    return rmse

# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best RMSE:", study.best_value)
print("Best params:", study.best_params)


# %% [markdown]
#  # train + val into full dataset

# %% [code]
import pandas as pd

# Concatenate back train + val into full dataset
X_full = pd.concat([X_train, X_val], axis=0)
y_full = pd.concat([y_train, y_val], axis=0)


# %% [markdown]
# # Get best params from Optuna

# %% [code] {"scrolled":true}
# 1. Get best params from Optuna
best_params = study.best_params

# 2. Retrain on full train data (combine train+val)
best_model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1)

best_model.fit(X_full, y_full, eval_metric="rmse")

# 3. Predict on X_test
y_test_pred = best_model.predict(X_test)

print("Test predictions:", y_test_pred[:10])  # show first 10 predictions


# %% [markdown]
# # submission of lgbm using optuna 

# %% [code]
# Choose one of the models for submission or average their predictions
# Example using LGBM predictions:
submission_df = pd.DataFrame({'id': test['id'], 'BeatsPerMinute':y_test_pred})
# Example using averaged predictions (if you want to try ensembling):
# averaged_predictions = (lgbm_test_pred + xgb_test_pred + catboost_test_pred) / 3
# submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': averaged_predictions})


# Save the submission file
submission_df.to_csv('submissionb.csv', index=False)

print("Submission file created successfully!")

# %% [markdown]
# # Model : 1

# %% [markdown]
# # LGBM model (basic)

# %% [markdown]
#  **LightGBM (Basic)**
#    - Used predefined parameters without tuning.  
#    - Served as a baseline LightGBM model.  
#    - Predictions saved as `submission.csv`.

# %% [code]
# Initialize the LGBM model
lgbm = LGBMRegressor(learning_rate =  0.02493303813836915,num_leaves = 34,max_depth = 13,reg_alpha =0.07319459385373907,reg_lambda = 0.08870554420170793,subsample= 0.5514983899131313,random_state=42)

# Train the LGBM model
print("Training LGBM model...")
lgbm.fit(X_train, y_train)

# Evaluate the LGBM model
y_pred_lgbm = lgbm.predict(X_val)
rmse_lgbm = mean_squared_error(y_val, y_pred_lgbm)**0.5
print(f"LGBM RMSE on validation data: {rmse_lgbm}")

# %% [markdown]
# # Model : 2

# %% [markdown]
# # XGBoost model

# %% [markdown]
#  **XGBoost**
#    - Trained an **XGBRegressor** with default parameters.  
#    - Evaluated RMSE on validation data.  
#    - Predictions saved as `submission2.csv`.

# %% [code]
# Initialize the XGBoost model
xgb = XGBRegressor(random_state=42)

# Train the XGBoost model
print("Training XGBoost model...")
xgb.fit(X_train, y_train)

# Evaluate the XGBoost model
y_pred_xgb = xgb.predict(X_val)
rmse_xgb = mean_squared_error(y_val, y_pred_xgb)**0.5
print(f"XGBoost RMSE on validation data: {rmse_xgb}")

# %% [markdown]
# # Model : 3

# %% [markdown]
# # catboost model

# %% [markdown]
# **CatBoost**
#    - Trained a **CatBoostRegressor** with random_state = 42.  
#    - Evaluated RMSE on validation data.  
#    - Predictions saved as `submission3.csv`.

# %% [code]
!pip install catboost

# %% [code]
from catboost import CatBoostRegressor

# Initialize the CatBoost model
catboost = CatBoostRegressor(random_state=42, verbose=0) # verbose=0 to suppress output during training

# Train the CatBoost model
print("Training CatBoost model...")
catboost.fit(X_train, y_train)

# Evaluate the CatBoost model
y_pred_catboost = catboost.predict(X_val)
rmse_catboost = mean_squared_error(y_val, y_pred_catboost)**0.5
print(f"CatBoost RMSE on validation data: {rmse_catboost}")

# %% [code] {"execution":{"iopub.status.busy":"2025-09-21T17:12:27.601515Z","iopub.execute_input":"2025-09-21T17:12:27.602155Z","iopub.status.idle":"2025-09-21T17:12:28.117187Z","shell.execute_reply.started":"2025-09-21T17:12:27.602130Z","shell.execute_reply":"2025-09-21T17:12:28.116495Z"}}
# Make predictions on the test data using the trained models
lgbm_test_pred = lgbm.predict(X_test)
xgb_test_pred = xgb.predict(X_test)
catboost_test_pred = catboost.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2025-09-21T17:12:32.467127Z","iopub.execute_input":"2025-09-21T17:12:32.467677Z","iopub.status.idle":"2025-09-21T17:12:32.956126Z","shell.execute_reply.started":"2025-09-21T17:12:32.467652Z","shell.execute_reply":"2025-09-21T17:12:32.955402Z"}}
# Choose one of the models for submission or average their predictions
# Example using LGBM predictions:
submission_df = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': lgbm_test_pred})
# Example using averaged predictions (if you want to try ensembling):
# averaged_predictions = (lgbm_test_pred + xgb_test_pred + catboost_test_pred) / 3
# submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': averaged_predictions})


# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")

# %% [code]
# Choose one of the models for submission or average their predictions
# Example using LGBM predictions:
submission_df = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': xgb_test_pred})
# Example using averaged predictions (if you want to try ensembling):
# averaged_predictions = (lgbm_test_pred + xgb_test_pred + catboost_test_pred) / 3
# submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': averaged_predictions})


# Save the submission file
submission_df.to_csv('submission2.csv', index=False)

print("Submission file created successfully!")

# %% [code]
# Choose one of the models for submission or average their predictions
# Example using LGBM predictions:
submission_df = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': catboost_test_pred})
# Example using averaged predictions (if you want to try ensembling):
# averaged_predictions = (lgbm_test_pred + xgb_test_pred + catboost_test_pred) / 3
# submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': averaged_predictions})


# Save the submission file
submission_df.to_csv('submission3.csv', index=False)

print("Submission file created successfully!")

# %% [markdown]
# ## 🏁 Submission
# - Multiple submission files were generated:
#   - `submissionb.csv` → LGBM with Optuna  
#   - `submission.csv` → Basic LGBM  
#   - `submission2.csv` → XGBoost  
#   - `submission3.csv` → CatBoost  

# %% [markdown]
# ## ✅ Conclusion
# - LightGBM with Optuna tuning achieved the best validation performance.

# %% [code]
