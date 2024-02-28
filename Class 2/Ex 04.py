# Data Engineering
# NYU Tandon School of Engineering
# Class 2.2 - Data Engineering Best Practices
# Fall 2023
# Prof. Carlos De Oliveira - Carlos.DeOliveira@NYU.edu 

# Import modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# function to get toy moon set
def get_moon_data():
    # make blobs and split into train and test sets
    X_, y_ = make_moons(n_samples=150, noise=0.4, random_state=42)
    X_train_, X_test_, y_train_, y_test_ = \
        train_test_split(X_, y_, test_size=.33, random_state=42)

    return [X_train_, X_test_, y_train_, y_test_]

### Support Vector Machine Classification ###
# import modules
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# get moon dataset
X_train_, X_test_, y_train_, y_test_ = get_moon_data()

print(X_train_)
print(y_train_)
print(X_test_)
print(y_test_)

#instantiate classifier object and fit to training data
clf = SVC(kernel="linear", C=0.5)
clf.fit(X_train_, y_train_)

# predict on test set and score the predictions against y_test
y_pred = clf.predict(X_test_)
f1 = f1_score(y_test_, y_pred) 
print('f1 score (SVM) is = ' + str(f1))


### SVM with Gaussian Kernel Classification ###
# instantiate classifier object and fit to training data
clf = SVC(gamma=2, C=1)
clf.fit(X_train_, y_train_)

# predict on test set and score the predictions against y_test
y_pred = clf.predict(X_test_)
f1 = f1_score(y_test_, y_pred) 
print('f1 score (SVM with Kernel) is = ' + str(f1))




### Decision Tree Classification ###
# import modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# get moon dataset
X_train, X_test, y_train, y_test = get_moon_data()

#instantiate classifier object and fit to training data
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# predict on test set and score the predictions against y_test
y_pred = clf.predict(X_test)
f1 = f1_score(y_test, y_pred) 
print('f1 score (Decision Tree) is = ' + str(f1))




### Random Forest Classification ###
# import modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# get moon dataset
X_train, X_test, y_train, y_test = get_moon_data()

#instantiate classifier object and fit to training data
clf = RandomForestClassifier(max_depth=4, n_estimators=4, 
                             max_features='sqrt', random_state=42)
clf.fit(X_train, y_train)

# predict on test set and score the predictions against y_test
y_pred = clf.predict(X_test)
f1 = f1_score(y_test, y_pred) 
print('f1 score (Random Forest) is = ' + str(f1))




### Cross Validation ###
# load iris and create X and y
from sklearn.datasets import load_iris
dataset = load_iris()
X,y = dataset.data, dataset.target

# import module
from sklearn.model_selection import train_test_split

# create train and test sets
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.33)

# create validation set from training set
X_train, X_val, y_train, y_val = \
        train_test_split(X, y, test_size=.33)




### k-fold Cross Validation ###
# load iris and create X and y
from sklearn.datasets import load_iris
dataset = load_iris()
X,y = dataset.data, dataset.target

# import modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

# create train and test sets
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.33)

# instantiate classifier object and pass to cross_val_score function
clf = LogisticRegression(solver='lbfgs', multi_class='ovr')
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')
print(scores)




### Grid Search with k-fold Cross-validation ###
# load iris and create X and y
from sklearn.datasets import load_iris
dataset = load_iris()
X,y = dataset.data, dataset.target

# import modules
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split

# create train and test sets
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.33)

# instantiate svc and gridsearch object and fit 
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 5, 10]}
svc = SVC(gamma='auto')
clf = GridSearchCV(svc, parameters, cv=5, scoring='f1_macro')
clf.fit(X_train, y_train)

# print best scoring classifier
print('Best score is = ' + str(clf.best_score_))
print('Best parameters are = ' + str(clf.best_params_))













### Building a Pipeline ###

# We will build a pipeline that transforms the data with PCA 
# and then predicts labels with LogisticRegression. 
# Let's start by loading the iris dataset and required modules, 
# and splitting the data into a train/test set. 
# We will use k-fold cross-validation in the grid search, 
# so no need to make a separate validation set.

# load iris and create X and y
from sklearn.datasets import load_iris
dataset = load_iris()
X,y = dataset.data, dataset.target

# import modules 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# create train and test sets
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.33)
        
# Now, we can instantiate the transformer and classifier objects, 
# and feed them into the pipeline.
# instantiate the transformer and classifier objects
pca = PCA()
logistic = LogisticRegression(solver='liblinear', multi_class='ovr', C=1.5)

# instantiate a pipeline and add steps to the pipeline
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

# Next, we will create the parameter grid that the grid search will use 
# and instantiate the grid search object. 
# Here, we will test a few values of n_components for PCA and C 
# for logistic regression using 5-fold cross-validation. 
# Finally, we fit our model to the data and print out the best parameters. 

# set the parameter grid to be passed to the grid search
param_grid = {
    'pca__n_components': [2, 3, 4],
    'logistic__C': [0.5, 1, 5, 10],
}

# instantiate the grid search object and pass the pipe and param_grid
model = GridSearchCV(pipe, param_grid, cv=5,
                      return_train_score=False)

# fit entire pipeline using grid search and 5-fold cross validation
model.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % model.best_score_)
print(model.best_params_)

# use the resulting pipeline to predict on new data
y_pred = model.predict(X_test)
print(X_test)
print(y_pred)





















