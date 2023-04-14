import numpy as np
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load iris dataset
iris_data = load_iris()
print(iris_data)

# create min-max scaler object
scaler_min_max = MinMaxScaler()

# perform min-max normalization on the data
iris_data_normalized = scaler_min_max.fit_transform(iris_data.data)

# train-test split the normalized data
X_train, X_test, y_train, y_test = train_test_split(iris_data_normalized, iris_data.target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with min-max normalization: {acc}")

# initialize StandardScaler object
scaler_Z = StandardScaler()

# perform z-normalization on the data
iris_data_normalized = scaler_Z.fit_transform(iris_data.data)

# train-test split the normalized data
X_train, X_test, y_train, y_test = train_test_split(iris_data_normalized, iris_data.target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with z-normalization: {acc}")

# calculate number of NaN values based on %
nan_percent = 5
n_samples, n_features = iris_data.data.shape
n_nan = int(n_samples * n_features * nan_percent / 100)

# create random indices for NaN values
nan_index = np.random.choice(n_samples * n_features, n_nan, replace=False)

# set selected values to NaN
iris_data.data.ravel()[nan_index] = np.nan

# count number of NaN values and print
nan_count = np.isnan(iris_data.data).sum()
print(f"Number of NaN values created: {nan_count} ({nan_percent}% of the dataset)")

# method1: ignore rows containing missing data
ignore_rows_data = iris_data.data[~np.isnan(iris_data.data).any(axis=1)]
ignore_rows_target = iris_data.target[~np.isnan(iris_data.data).any(axis=1)]


# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(ignore_rows_data, ignore_rows_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with ignore rows method: {acc}")


# method 2: fill in manually
manual_data = iris_data.data.copy()
manual_target = iris_data.target.copy()

# replace NaN with mean of the corresponding feature
for i in range(n_features):
    mean_val = np.mean(manual_data[:, i][~np.isnan(manual_data[:, i])])
    manual_data[:, i][np.isnan(manual_data[:, i])] = mean_val



# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(manual_data, manual_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with manual fill in method: {acc}")

# method 3: use a global constant
constant_data = iris_data.data.copy()
constant_target = iris_data.target.copy()

# replace NaN with a constant value (zero in this example)
constant_value = 0
constant_data[np.isnan(constant_data)] = constant_value

# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(constant_data, constant_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with global constant in method: {acc}")


# method 4 : fill with measure of central tendency
central_data = iris_data.data.copy()
central_target = iris_data.target.copy()

# replace NaN with mean value of the corresponding feature
for i in range(n_features):
    central_data[:, i][np.isnan(central_data[:, i])] = np.mean(central_data[:, i][~np.isnan(central_data[:, i])])

# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(central_data, central_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with fill measure of central tendency in method: {acc}")

# method 5: fill with most probable value
probable_data = iris_data.data.copy()
probable_target = iris_data.target.copy()

# use SimpleImputer to fill NaN values with the most frequent value of the corresponding feature
imp = SimpleImputer(strategy="most_frequent")
probable_data = imp.fit_transform(probable_data)

# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(probable_data, probable_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with Use the most probable value in method: {acc}")


# calculate number of NaN values based on %
nan_percent = 10
n_samples, n_features = iris_data.data.shape
n_nan = int(n_samples * n_features * nan_percent / 100)

# create random indices for NaN values
nan_index = np.random.choice(n_samples * n_features, n_nan, replace=False)

# set selected values to NaN
iris_data.data.ravel()[nan_index] = np.nan

# count number of NaN values and print
nan_count = np.isnan(iris_data.data).sum()
print(f"Number of NaN values created: {nan_count} ({nan_percent}% of the dataset)")

# method1: ignore rows containing missing data
ignore_rows_data = iris_data.data[~np.isnan(iris_data.data).any(axis=1)]
ignore_rows_target = iris_data.target[~np.isnan(iris_data.data).any(axis=1)]


# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(ignore_rows_data, ignore_rows_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with ignore rows method: {acc}")


# method 2: fill in manually
manual_data = iris_data.data.copy()
manual_target = iris_data.target.copy()

# replace NaN with mean of the corresponding feature
for i in range(n_features):
    mean_val = np.mean(manual_data[:, i][~np.isnan(manual_data[:, i])])
    manual_data[:, i][np.isnan(manual_data[:, i])] = mean_val



# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(manual_data, manual_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with manual fill in method: {acc}")

# method 3: use a global constant
constant_data = iris_data.data.copy()
constant_target = iris_data.target.copy()

# replace NaN with a constant value (zero in this example)
constant_value = 0
constant_data[np.isnan(constant_data)] = constant_value

# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(constant_data, constant_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with global constant in method: {acc}")


# method 4 : fill with measure of central tendency
central_data = iris_data.data.copy()
central_target = iris_data.target.copy()

# replace NaN with mean value of the corresponding feature
for i in range(n_features):
    central_data[:, i][np.isnan(central_data[:, i])] = np.mean(central_data[:, i][~np.isnan(central_data[:, i])])

# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(central_data, central_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with fill measure of central tendency in method: {acc}")

# method 5: fill with most probable value
probable_data = iris_data.data.copy()
probable_target = iris_data.target.copy()

# use SimpleImputer to fill NaN values with the most frequent value of the corresponding feature
imp = SimpleImputer(strategy="most_frequent")
probable_data = imp.fit_transform(probable_data)

# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(probable_data, probable_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with Use the most probable value in method: {acc}")

# calculate number of NaN values based on %
nan_percent = 15
n_samples, n_features = iris_data.data.shape
n_nan = int(n_samples * n_features * nan_percent / 100)

# create random indices for NaN values
nan_index = np.random.choice(n_samples * n_features, n_nan, replace=False)

# set selected values to NaN
iris_data.data.ravel()[nan_index] = np.nan

# count number of NaN values and print
nan_count = np.isnan(iris_data.data).sum()
print(f"Number of NaN values created: {nan_count} ({nan_percent}% of the dataset)")

# method1: ignore rows containing missing data
ignore_rows_data = iris_data.data[~np.isnan(iris_data.data).any(axis=1)]
ignore_rows_target = iris_data.target[~np.isnan(iris_data.data).any(axis=1)]


# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(ignore_rows_data, ignore_rows_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with ignore rows method: {acc}")


# method 2: fill in manually
manual_data = iris_data.data.copy()
manual_target = iris_data.target.copy()

# replace NaN with mean of the corresponding feature
for i in range(n_features):
    mean_val = np.mean(manual_data[:, i][~np.isnan(manual_data[:, i])])
    manual_data[:, i][np.isnan(manual_data[:, i])] = mean_val



# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(manual_data, manual_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with manual fill in method: {acc}")

# method 3: use a global constant
constant_data = iris_data.data.copy()
constant_target = iris_data.target.copy()

# replace NaN with a constant value (zero in this example)
constant_value = 0
constant_data[np.isnan(constant_data)] = constant_value

# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(constant_data, constant_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with global constant in method: {acc}")


# method 4 : fill with measure of central tendency
central_data = iris_data.data.copy()
central_target = iris_data.target.copy()

# replace NaN with mean value of the corresponding feature
for i in range(n_features):
    central_data[:, i][np.isnan(central_data[:, i])] = np.mean(central_data[:, i][~np.isnan(central_data[:, i])])

# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(central_data, central_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with fill measure of central tendency in method: {acc}")

# method 5: fill with most probable value
probable_data = iris_data.data.copy()
probable_target = iris_data.target.copy()

# use SimpleImputer to fill NaN values with the most frequent value of the corresponding feature
imp = SimpleImputer(strategy="most_frequent")
probable_data = imp.fit_transform(probable_data)

# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(probable_data, probable_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with Use the most probable value in method: {acc}")

# calculate number of NaN values based on %
nan_percent = 20
n_samples, n_features = iris_data.data.shape
n_nan = int(n_samples * n_features * nan_percent / 100)

# create random indices for NaN values
nan_index = np.random.choice(n_samples * n_features, n_nan, replace=False)

# set selected values to NaN
iris_data.data.ravel()[nan_index] = np.nan

# count number of NaN values and print
nan_count = np.isnan(iris_data.data).sum()
print(f"Number of NaN values created: {nan_count} ({nan_percent}% of the dataset)")

# method1: ignore rows containing missing data
ignore_rows_data = iris_data.data[~np.isnan(iris_data.data).any(axis=1)]
ignore_rows_target = iris_data.target[~np.isnan(iris_data.data).any(axis=1)]


# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(ignore_rows_data, ignore_rows_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with ignore rows method: {acc}")


# method 2: fill in manually
manual_data = iris_data.data.copy()
manual_target = iris_data.target.copy()

# replace NaN with mean of the corresponding feature
for i in range(n_features):
    mean_val = np.mean(manual_data[:, i][~np.isnan(manual_data[:, i])])
    manual_data[:, i][np.isnan(manual_data[:, i])] = mean_val



# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(manual_data, manual_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with manual fill in method: {acc}")

# method 3: use a global constant
constant_data = iris_data.data.copy()
constant_target = iris_data.target.copy()

# replace NaN with a constant value (zero in this example)
constant_value = 0
constant_data[np.isnan(constant_data)] = constant_value

# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(constant_data, constant_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with global constant in method: {acc}")


# method 4 : fill with measure of central tendency
central_data = iris_data.data.copy()
central_target = iris_data.target.copy()

# replace NaN with mean value of the corresponding feature
for i in range(n_features):
    central_data[:, i][np.isnan(central_data[:, i])] = np.mean(central_data[:, i][~np.isnan(central_data[:, i])])

# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(central_data, central_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with fill measure of central tendency in method: {acc}")

# method 5: fill with most probable value
probable_data = iris_data.data.copy()
probable_target = iris_data.target.copy()

# use SimpleImputer to fill NaN values with the most frequent value of the corresponding feature
imp = SimpleImputer(strategy="most_frequent")
probable_data = imp.fit_transform(probable_data)

# train-test split the data
X_train, X_test, y_train, y_test = train_test_split(probable_data, probable_target, test_size=0.2, random_state=42)

# train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# make predictions on test data and calculate accuracy
y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gaussian Naive Bayes classifier with Use the most probable value in method: {acc}")









