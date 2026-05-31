# traffic_demand_prediction
This repo is made to document and maintain all the versions of ML Model made for GridLock Hackathon 2.0

Files Overview:

1. Dataset Folder: contains all the datasets provided for the problem like training data as train.csv, testing dataset as test.csv etc.

Our goal is to make a ML Model and train and evaluate it using train.csv dataset and make predictions on test.csv dataset.

Read README.md in the Dataset Folder to gain more insights about the columns of the dataset.

2. Submissions Folder: contains all the submission made read README.md in the Submissions Folder to get the scores obtained by the respective submission file.

3. train_demand_prediction.ipynb: it is python notebook used to make ML Model both 'submission01.csv' and 'submission02.csv' are made using different versions of this notebook using the "Approach01". The current version is more tuned version and was used to make 'submission02.csv' which produces better 'score'.

4. train_demand_prediction_(1).ipynb: it is python notebook used to make ML Model all submissions starting from 'submission03.csv' are made using this versions of this notebook using the "Approach02". The current version is one that produced the best 'score' with 'submission05.csv'.

Note: Both "Appoach01" and "Approach02" are mensioned bellow read then for details about 'Feature Engineering', 'Model Tuning' and other changes done to the approach.

# "Approach01":

### Feature Engineering:

1. geohash: We studied the data in 'geohash' manually and found that since we are only dealing with datapoints from a city ('bangaluru' in this problem) the first three digits of 'geohash' are the same. So, we will split the last three digits into features like 'geohash_location_0', 'geohash_location_1' and 'geohash_location_2'.

2. timestamp: This data only contains 'hour' and 'minutes' so we split this column into two columns called 'hour' and 'minutes'.

# "Approach02":

### Feature Engineering:

1. geohash: Unlike 'Approach01', we didn't split the last three digits of 'geohash' but used the last three digits together to denote unique location called 'geohash_location'.

2. timestamp: This data only contains 'hour' and 'minutes' so we split this column into two columns called 'hour' and 'minutes'.

### Difference in both Approaches:

The key difference in both approaches is the different way of utilising the 'geohash' column. 'Approach01' results in much lower number of input features for model (around 92), while 'Approach02' increase the input feature size exponantially to around 2572 features.

### Procedure:

The remaining procedure is similar for both approaches.

#### Model: XGBoost Regressor (Tuned using GridSearchCV)

#### Dealing with null values:

For columns like 'RoadType' and 'Weather' we fill the null values with 'Unknown'.

For numerial columns like 'Temperature' we fill the value with the mean of the data grouped by 'Weather'.

#### Converting Non-Numerical Data into Numbers:

Columns like 'LargeVehicles' and 'Landmarks' contains boolean values in form of 'Allowed', 'Not Allowed', 'Yes', 'No' etc. So, we replace the respective boolean values with 1 and 0 to make them numerical.

Then, we one-hot encode the categorical columns using 'get_dummies' function.

#### Training, Evaluating and Tuning Model
