# Data Engineering
# NYU Tandon School of Engineering
# Class 3.1 - Data Engineering Best Practices
# Fall 2023
# Prof. Carlos De Oliveira - Carlos.DeOliveira@NYU.edu 

# Best Practice 4:
# In scikit-learn, the Imputer class provides a nicely written imputation transformer. 
# We herein use it for the following small example:
import numpy as np
from sklearn.impute import SimpleImputer
data_origin = [[7, 2, 3], 
               [4, np.nan, 6], 
               [8, 3, 1], 
               [10, 7, 9]]
# Initialize the imputation transformer with the mean value and obtain such information from the original data:
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(data_origin)
# Complete the missing value as follows:
data_origin_mean=print(imp_mean.transform(data_origin))
# Similarly, initialize the imputation transformer with the median value, as detailed in the following:
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
imp_median.fit(data_origin)
# Complete the missing value as follows:
data_origin_median=print(imp_median.transform(data_origin))
# When new samples come in, the missing values (in any attribute) can be imputed 
# using the trained transformer, for example, with the mean value, as shown here:
new = [[20, np.nan, 5],
       [30, np.nan, 18],
       [np.nan, 70, 1],
       [np.nan, np.nan, 13]]
new_mean_imp = imp_mean.transform(new)
print(new_mean_imp)

# Now that we have seen how imputation works as well as its implementation, 
# let's explore how the strategy of imputing missing values and discarding 
# missing data affects the prediction results through the following example:
# 1. First we load the diabetes dataset and simulate a corrupted dataset with missing values, as shown here:
import numpy as np
from sklearn import datasets
dataset = datasets.load_diabetes()
X_full, y = dataset.data, dataset.target
# 2. Simulate a corrupted dataset by adding 25% missing values:
m, n = X_full.shape
m_missing = int(m * 0.25)
print(m, m_missing)
# 3. Randomly select the m_missing samples, as follows:
np.random.seed(42)
missing_samples = np.array([True] * m_missing + [False] * (m - m_missing))
np.random.shuffle(missing_samples)
# 4. For each missing sample, randomly select 1 out of n features:
missing_features = np.random.randint(low=0, high=n, size=m_missing)
# 5. Represent missing values by nan, as shown here:
X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = np.nan
# 6. Then we deal with this corrupted dataset by discarding the samples containing a missing value:
X_rm_missing = X_missing[~missing_samples, :]
y_rm_missing = y[~missing_samples]
# 7. Measure the effects of using this strategy by estimating the averaged 
# regression score, R2, with a regression forest model in a cross-validation manner. 
# Estimate R2 on the dataset with the missing samples removed, as follows:
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
regressor = RandomForestRegressor(random_state=42, max_depth=10, n_estimators=100)
score_rm_missing = cross_val_score(regressor, X_rm_missing, y_rm_missing, cv=5).mean()
print('Score with the data set with missing samples removed: {0:.2f}'.format(score_rm_missing))
# 8. Now we approach the corrupted dataset differently by imputing missing values with the mean:
# imp_mean = Imputer(missing_values='NaN', strategy='mean')
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
# X_mean_imp = imp_mean.fit_transform(X_missing)
X_mean_imp = imp_mean.fit(X_missing)
# 9. Similarly, measure the effects of using this strategy by estimating the averaged R2, as follows:
regressor = RandomForestRegressor(random_state=42, max_depth=10, n_estimators=500)
score_full = cross_val_score(regressor, X_full, y, cv=5).mean()
print('Score with the full data set: {0:.2f}'.format(score_full))
# 10. An imputation strategy works better than discarding in this case. So, how far 
# is the imputed dataset from the original full one? We can check it again by 
# estimating the averaged regression score on the original dataset, as follows:
regressor = RandomForestRegressor(random_state=42, max_depth=10, n_estimators=500)
score_full = cross_val_score(regressor, X_full, y, cv=5).mean()
print('Score with the full data set: {0:.2f}'.format(score_full))
# It turns out that little information is comprised in the completed dataset.
# However, there is no guarantee that an imputation strategy always works better, 
# and sometimes dropping samples with missing values can be more effective. 
# Hence, it is a great practice to compare the performances of different strategies 
# via cross-validation as we have done previously.


# Best Practice 8:
# The benefits of feature selection include the following:
# 1. Reducing the training time of prediction models, as redundant, or irrelevant features are eliminated
# 2. Reducing overfitting for the preceding same reason
# 3. Likely improving performance as prediction models will learn from data with more significant features
# For example, by executing the following steps, we can measure the effects of 
# feature selection by estimating the averaged classification accuracy with 
# an SVC model in a cross-validation manner:
# 1. First, we load the handwritten digits dataset from scikit-learn:
from sklearn.datasets import load_digits
dataset = load_digits()
X, y = dataset.data, dataset.target
print(X.shape)
# 2. Next, estimate the accuracy of the original dataset, which is 64 dimensional:
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
classifier = SVC(gamma=0.005)
score = cross_val_score(classifier, X, y, cv=5).mean()
print('Score with the original data set: {0:.2f}'.format(score))
# 3. Then conduct feature selection based on random forest and sort the features based on their importance scores:
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=-1)
random_forest.fit(X, y)
feature_sorted = np.argsort(random_forest.feature_importances_)
# 4.Now select a different number of top features to construct a new dataset, 
# and estimate the accuracy on each dataset:
K = [10, 15, 25, 35, 45]
for k in K:
    top_K_features = feature_sorted[-k:]
    X_k_selected = X[:, top_K_features]
    # Estimate accuracy on the data set with k selected features
    classifier = SVC(gamma=0.005)
    score_k_features = cross_val_score(classifier, X_k_selected, y, cv=5).mean()
    print('Score with the data set of top {0} features: {1:.2f}'.format(k, score_k_features))


# Best Practice 9:
# It is not guaranteed that dimensionality reduction will yield 
# better prediction results - let's go over an example. 
from sklearn.decomposition import PCA
# Keep different number of top components
N = [10, 15, 25, 35, 45]
for n in N:
    pca = PCA(n_components=n)
    X_n_kept = pca.fit_transform(X)
    # Estimate accuracy on the data set with top n components
    classifier = SVC(gamma=0.005)
    score_n_components = cross_val_score(classifier, X_n_kept, y, cv=5).mean()
    print('Score with the data set of top {0} components: {1:.2f}'.format(n, score_n_components))


# Best Practice 12:
# Performing feature engineering without domain expertise - some generic approaches
# 1. Binarization: This is the process of converting a numerical feature to a
# binary one with a preset threshold. For example, the feature "number of visits
# per week" can be used to produce a new feature by judging whether the value 
# is greater than or equal to 3. We implement such binarization using scikit-learn:
from sklearn.preprocessing import Binarizer
X = [[4], [1], [3], [0]]
binarizer = Binarizer(threshold=2.9)
X_new = binarizer.fit_transform(X)
print(X_new)
# 2. Discretization: This is the process of converting a numerical feature to a 
# categorical feature with limited possible values. Binarization can be viewed as
# a special case of discretization. For example, we can generate an age group feature:
# "18-24" for age from 18 to 24, "25-34" for age from 25 to 34, "34-54", and "55+".
# 3. Interaction: This includes the sum, multiplication, or any operations of two 
# numerical features, joint condition check of two categorical features. For example, 
# the number of visits per week and the number of products purchased per week can be 
# used to generate the number of products purchased per visit feature; interest and 
# occupation, such as sports and engineer, can form occupation AND interest, 
# such as engineer interested in sports.
# 4. Polynomial transformation: This is a process of generating polynomial and 
# interaction features. For two features, a and b, the two degree of polynomial 
# features generated are a2, ab, and b2. In scikit-learn, we can use the 
# PolynomialFeatures class to perform polynomial transformation: 
from sklearn.preprocessing import PolynomialFeatures
X = [[2, 4],
     [1, 3],
     [3, 2],
     [0, 3]]
poly = PolynomialFeatures(degree=2)
X_new = poly.fit_transform(X)
print(X_new)


# Best Practice 16:
dataset = datasets.load_diabetes()
X, y = dataset.data, dataset.target
num_new = 30 # the last 30 samples as new data set
X_train = X[:-num_new, :]
y_train = y[:-num_new]
X_new = X[-num_new:, :]
y_new = y[-num_new:]
# Preprocess the training data with scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
# Now save the established standardizer, the scaler object with pickle, which generates the scaler.p file
import pickle
pickle.dump(scaler, open("scaler.p", "wb" ))
# Move on with training a SVR model on the scaled data
X_scaled_train = scaler.transform(X_train)
from sklearn.svm import SVR
regressor = SVR(C=20)
regressor.fit(X_scaled_train, y_train)
# Save the trained regressor object with pickle, which generates a regressor.p file
pickle.dump(regressor, open("regressor.p", "wb"))
# In the deployment stage, we first load the saved standardizer and regressor object from the preceding two files
my_scaler = pickle.load(open("scaler.p", "rb" ))
my_regressor = pickle.load(open("regressor.p", "rb"))
# Then preprocess the new data using the standardizer and make prediction with the regressor object just loaded
X_scaled_new = my_scaler.transform(X_new)
predictions = my_regressor.predict(X_scaled_new)



