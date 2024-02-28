# Data Engineering
# NYU Tandon School of Engineering
# Class 1.2 - Ten steps: preparing data
# Summer 2023
# Prof. Carlos De Oliveira - Carlos.DeOliveira@NYU.edu 

#################################

# 1. First step: connecting to a DB
import sqlite3
sqlite_file = 'C:/Users/carlo/OneDrive/Documents/Documentos/CJ/NYU Teaching/Data Engineering/Codes/data/boston.db' 

# connecting to the database file
conn = sqlite3.connect(sqlite_file)

# initialize a cursor obect
cur = conn.cursor()

# define a traversing search
cur.execute("select * from boston limit 100;")

# fetch and print
data = cur.fetchall()
print(data)

# For our example, we use the included database (.db) file in the place of 
# an actual remote database. This means that we will connect to this file. 
# In practice, you will connect to a remote location by using a network address and login credentials.


#################################


# 2. Second step: SQL
cur.execute("select ZN from boston where ZN > 0.0;")
data = cur.fetchall()
print(data)

# Now, let's first duplicate our original query for the entire table, which was limited to 100 rows. 
import pandas as pd
# get all data inside boston table limited to 100 rows
df = pd.read_sql_query("select * from boston limit 100;", conn)
print("df.shape = " + str(df.shape))

# Now, let's grab the whole table and use pandas to sanity check with the head() method and print the summary
df = pd.read_sql_query("select * from boston;", conn)
print("df.shape = " + str(df.shape))
print("Sanity check with Pandas head():")
print(df.head())
print("Summarize with Pandas describe():")
print(df.describe().transpose())

# Let's continue with a new helpful example: get all data inside boston table that has ZN values greater 0
df = pd.read_sql_query("select * from boston where ZN > 0.0;", conn)
print("df.shape = " + str(df.shape))

# Same as above with additional filtering: records greater than 250
df = pd.read_sql_query("select * from boston where ZN > 0.0 and record > 250;", conn)
print("df.shape = " + str(df.shape))

# This is an example of multiline search syntax
df = pd.read_sql_query("""
                       select record, ZN, AGE, TAX from boston
                       where ZN > 0.0 and CRIM < 2.5;
                       """,
                       conn)

# use Pandas 'to_sql' method to commit changes to connection
df.to_sql("C:/Users/carlo/OneDrive/Documents/Documentos/CJ/NYU Teaching/Data Engineering/Codes/data/boston_updated", conn, if_exists="replace")
# close connection
conn.close()

#################################


# 3. Third step: change and save to local disks
# load from file
df = pd.read_csv("C:/Users/carlo/OneDrive/Documents/Documentos/CJ/NYU Teaching/Data Engineering/Codes/data/iris.csv")
# make some changes
# create an index
df.index.name = "record"
df['species'] = "new-species"
print(df.head())

# save to file
df.to_csv("C:/Users/carlo/OneDrive/Documents/Documentos/CJ/NYU Teaching/Data Engineering/Codes/data/iris_updated.csv", index=True) 

# load from web URL
# url = "www.nyu.edu\blablabla.csv (or .data, etc.)

#################################


# 4. 4th step: access, search, and sanity checks

# Before we move on, first load the data from the included boston.db file, as follows:
import pandas as pd
import sqlite3
sqlite_file = 'C:/Users/carlo/OneDrive/Documents/Documentos/CJ/NYU Teaching/Data Engineering/Codes/data/boston.db' 
# Connecting to the database file
conn = sqlite3.connect(sqlite_file)

df = pd.read_sql_query("select * from boston;", conn)
print("df.shape = " + str(df.shape))
df.set_index("record", inplace=True)
conn.close()

# Some more sanity checks: print first 5 rows with column names
print(df.head())

# Let's use some more sanity checks to understand more about the data: 
# get amount of rows and columns
print(df.shape)
# get columns in the dataframe
print(df.columns)

# Now, use the .describe() method to easily get some summary statistics:
# get statistical summary
df.describe()
# view in transposed form
print(df.describe().transpose())

# Let's use .min(), .max(), .mean(), and .median() methods as well:
# get max and min values
df.max()
df.min()
df.mean()
print(df.median())

# Now, let's get the index of the maximum and minimum values using the .idmax() and .idmin():
# get index of max and min values
df.idxmax()
print(df.idxmin())

# Additionally, we can get specific rows with ease:
# get first row of data (index=0)
df.loc[0]
# get third row of data (index=2)
df.loc[2]
# print(df.loc[2])
# get first row of CRIM column
print(df.loc[0]['CRIM'])

# We can also isolate single columns, as follows:
# isolate single columns
df['AGE'].mean()
df['MEDV'].idxmax()
print(df['AGE'].idxmin())

# We can even sort according to specific columns with the byarg.
# sort (ascending by default)
df.sort_values(by = 'ZN')
# sort descending
df.sort_values(by = 'ZN', ascending = False)
print(df.sort_values(by = 'ZN', ascending = False).head())

# Now, let's do a permanent sort on the table with the inplace arg, 
# permanently changing how it's stored in memory:
# permanently sort the table
df.sort_values(by = 'ZN', inplace=True)
# now call df.head() on permanently sorted table
print(df.head())

# In case we change our minds and want to undo the permanent sort, 
# we can sort according to the original index column and get our original data back:
# sort back on index
df.sort_values(by = 'record', inplace=True)
print(df.head())

# As a final example, let's chain together a couple of filters 
# and use the .describe() method for some summary statistics of the smaller, 
# filtered dataset: 
# filter dataframe to show only even records
df[df.index % 2 == 0]
# filter dataframe to show only record with AGE greater than 95
df[df['AGE'] > 95]
# get statistical summary of the filtered table
df[df['AGE'] > 95].describe().transpose()

#################################


# 5. 5th step: basic plotting

# Scatter plot:
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
# load iris
df = pd.read_csv("C:/Users/carlo/OneDrive/Documents/Documentos/CJ/NYU Teaching/Data Engineering/Codes/data/iris.csv")
# scatter plot
sns.scatterplot(x='petal length in cm', y='petal width in cm', data=df)
# next, label the data points by color and add the legend with the hue argument:
sns.scatterplot(x='petal length in cm', y='petal width in cm', 
                hue='species', data=df) 
# next, scatter plot points using lmplot with fit_reg=False:
sns.lmplot(x='petal length in cm', y='petal width in cm', 
           hue="species", data=df, fit_reg=False,
           palette='bright',markers=['o','x','v'])

# Histograms
sns.distplot(df['petal length in cm'])
# histogram with 15 bins
sns.distplot(df['petal length in cm'], bins=15)

# Jointplots
sns.jointplot(x='petal length in cm', y='petal width in cm', 
              data=df, kind='scatter', marginal_kws=dict(bins=10))
# jointplot with kde
sns.jointplot(x='petal length in cm', y='petal width in cm', 
              data=df, kind='kde')

# Violin plots
# y='sepal width in cm'
sns.violinplot(x='species',y='sepal width in cm', data=df)
# y='petal width in cm'
sns.violinplot(x='species',y='petal width in cm', data=df)

# Pairplots
# Let's load up the boston dataset and build a pairplot with five selected variables:
import pandas as pd
import sqlite3
sqlite_file = 'C:/Users/carlo/OneDrive/Documents/Documentos/CJ/NYU Teaching/Data Engineering/Codes/data/boston.db' 
# Connecting to the database file
conn = sqlite3.connect(sqlite_file)
df = pd.read_sql_query("select * from boston;", conn)
conn.close()
sns.pairplot(data=df)
# pairplot with selected features
vars_to_plot = ['CRIM', 'AGE', 'DIS', 'LSTAT', 'MEDV']
sns.pairplot(data=df, vars=vars_to_plot)

#################################


# 7. 7th step: missing values
# load iris dataset with missing values with .isnull()
import pandas as pd
df = pd.read_csv("C:/Users/carlo/OneDrive/Documents/Documentos/CJ/NYU Teaching/Data Engineering/Codes/data/iris_missing_values.csv")
df.index.name = "record"
print(df.head())

# get boolean (True/False) response for each datapoint for NaNs 
df['sepal length in cm'].isnull()

# check if any missing values in column
print(df['sepal length in cm'].isnull().values.any())

# get number of many missing values in column
print(df['sepal length in cm'].isnull().values.sum())

# fill missing values with new values (--> "example"), store in new "df_example" dataframe 
df_example = df['sepal length in cm'].fillna('example')
print(df_example.head())

# drop rows with missing data
df_dropped = df.dropna(axis=0)
print(df_dropped.head())
# we can also drop columns in a similar fashion:
# df_dropped = df.dropna(axis=1)

# Imputing to replace the missing values
# In the case of intermittent missing values, 
# we can predict the replacement values for the empty cells. 
# The mathematical apparatus for predicting these values is called an imputor.

# import imputer module from Scikit-learn and instantiate imputer object
import numpy as np
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

#define columns to impute on
cols = ['sepal length in cm',
         'sepal width in cm',
         'petal length in cm',
         'petal width in cm',]

# fit imputer and transform dataset, store in df_new
out_imp = imputer.fit_transform(df[cols])
df_new = pd.DataFrame(data = out_imp, columns = cols)
df_new = pd.concat([df_new, df[['species']]], axis = 1)
print(df_new.head())

#################################


# 8. 8th step: scaling and normalization

# A mathematical property is considered to be scale-invariant if 
# it does not change when multiplying specified inputs by a constant.

# Scaling is important for transformation and learning algorithms that are 
# NOT scale-invariant. Two examples of algorithms that lack scale-invariance 
# are: (1) PCA and (2) penalized regression (latter in the course).

# load iris dataset
import numpy as np
df = pd.read_csv("C:/Users/carlo/OneDrive/Documents/Documentos/CJ/NYU Teaching/Data Engineering/Codes/data/iris.csv")
df.index.name = "record"

# define columns to scale
cols = ['sepal length in cm',
        'sepal width in cm',
        'petal length in cm',
        'petal width in cm']

# min-max normalization is one of the most popular scaling processes
# load module and instantiate scaler object
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# normalize the data and store in out_scaled numpy array
out_scaled = scaler.fit_transform(df[cols])
print(out_scaled)

# Standardization is another scaling process: it is used 
# to put the variation within each feature space. 
# Other values can be chosen for advanced standardization, 
# but variance=1 and mean=0 are the most common.
# load module and instantiate scaler object
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# standardize the data and store in out_scaled numpy array
out_scaled_ = scaler.fit_transform(df[cols])
print(out_scaled_)

#################################


# 9. 9th step: categorical data

# One consideration to be aware of is whether the variable is ordered or not. 
# For example, an athlete's shoe size is categorical and ordinal 
# because the larger shoe size does indicate a larger value, 
# whereas the shoe color is categorical, but not ordinal 
# because one color is not necessarily larger in value than another. 
# In the latter case, we call these variables nominal. 
# This section will introduce basic ordinal encoding 
# and a strategy called one-hot encoding, 
# which is commonly used for both ordinal and nominal variables. 
# It will end with a simple label encoding section for converting 
# categorical target variables into something useful in a short, single step.

# Before we begin the section, let's load our small example long jump dataset, as follows: 
# load example long jump dataset
df = pd.read_csv("C:/Users/carlo/OneDrive/Documents/Documentos/CJ/NYU Teaching/Data Engineering/Codes/data/long_jump.csv")
df.set_index('Person', inplace=True)

# Ordinal encoding
# Ordinal variables have an order to them. Our examples 
# from the long jump dataset are Jersey Size and Shoe Size.
# filter in categorical columns ("cats") for demonstration
cats = ['Jersey Size', 'Shoe Size']
print(df[cats])

# import module and instantiate enc object
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()

# Now, we will use scikit-learn's OrdinalEncoder module to encode our ordinal columns. 
# fit and transform in one call and print categories
out_enc = enc.fit_transform(df[cats])
print('identified categories:')
print(enc.categories_)
print('encoded data:')
print(out_enc)

# Next, we will simply overwrite the original columns in our DataFrame 
# with the newly-encoded continuous features:
# overwrite categorical features in original dataframe
df[cats] = out_enc
print(df.head())

# One-hot encoding
# The one-hot technique emerged from the electronics field as a way to record 
# the state of a machine by using simple binary methods (that is, 0's and 1's). 
# Significant shortcoming of one-hot encoding: non-extrapolation of new states not available in the source.

# import module and instantiate enc object
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)

# fit and transform in one call and print categories
out_enc = enc.fit_transform(df[cats])
new_cols = enc.get_feature_names(cats).tolist()
print(new_cols)

# create temporary dataframe "df_enc" for concatenation with original data
df_enc = pd.DataFrame(data = out_enc, columns = new_cols)
df_enc.index = df.index

# drop original columns and concat new encoded columns
df.drop(cats, axis=1, inplace=True)
df = pd.concat([df, df_enc], axis = 1)
print(df.columns)

# Label encoding
# Often, the only column that needs encoding is the label or output column. 
# For these situations, scikit-learn includes the simple LabelEncoder module that encodes a single column.
# import modules and instantiate enc object
from sklearn import preprocessing
enc = preprocessing.LabelEncoder()

# fit with integer labels and transform
out_enc = enc.fit_transform([1, 2, 5, 2, 4, 2, 5])
print(out_enc)

# fit with string labels and transform
out_enc = enc.fit_transform(["blue", "red", "blue", "green", "red", "red"])
print(out_enc)

#################################


# 10. 10th step: dimension reduction
# Due to the curse of dimensionality, a reduction of the number of  
# feature columns might be required before you can get any work done.
# There are two main strategies for reducing dimensions, as follows:
# Selection: Choose the best features and eliminate the others.
# Transformation: Create new features that summarize the combinations of the original ones.

# Selection: (1) feature filtering; or (2) wrapper methods. 

# (1) feature filtering: (i) variance threshold; or (ii) correlation coefficient.

# The variance threshold
#    1 - Prefit with no threshold.
#    2 - Analyze the variances.
#    3 - Choose the threshold.
#    4 - Refit with the chosen threshold.
# load iris dataset
df = pd.read_csv("C:/Users/carlo/OneDrive/Documents/Documentos/CJ/NYU Teaching/Data Engineering/Codes/data/iris.csv")
df.index.name = 'record'
# define columns to filter
cols = ['sepal length in cm',
        'sepal width in cm',
        'petal length in cm',
        'petal width in cm',]
# instantiate Scikit-learn object with no threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
# prefit object with df[cols]
selector.fit(df[cols])
# check feature variances before selection
print(selector.variances_)
# For demonstration purposes, we will choose 0.6 as the threshold and then refit. 
# From the output, you should expect columns 0 and 2 (0.68 and 3.09) to be selected.
# set threshold into selector object
selector.set_params(threshold=0.6)
# refit and transform, store output in out_sel
out_sel = selector.fit_transform(df[cols])
# check which features were chosen
print(selector.get_support())
# filter in the selected features
df_sel = df.iloc[:, selector.get_support()]
# add labels to new dataframe and sanity check
df_sel = pd.concat([df_sel, df[['species']]], axis = 1)
print(df_sel.head())

# The correlation coefficient
# import matplotlib for access to color maps
import matplotlib.pyplot as plt
# load boston dataset
from sklearn.datasets import load_boston
dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['MEDV'] = dataset.target; df.index.name = 'record'
# find correlation with pandas ".corr()"
cor = df.corr()
# visualize with Seaborn heat map, color map = Blues
sns.heatmap(cor, annot=False, cmap=plt.cm.Blues)
plt.show()
# get correlation values with target variable
cor_target = abs(cor['MEDV'])
print(cor_target)
# For demonstration purposes, we will choose 0.6 as the threshold and then filter. 
# From the output, you should expect columns 5 and 12 (0.69 and 0.74) to be selected.
# choose features above threshold 0.6
selected_cols = cor_target[cor_target>0.6]
print("selected columns, correlation with target > 0.6")
print(selected_cols)
# filter in the selected features
df_sel = df[selected_cols.index]
print(df_sel.head())

# (2) wrapper methods: 
# If we are willing to use the prediction algorithm's automatic scoring 
# (sequential feature selection), scikit-learn has a built-in method 
# called recursive feature elimination (RFE). 
# load iris dataset
df = pd.read_csv("C:/Users/carlo/OneDrive/Documents/Documentos/CJ/NYU Teaching/Data Engineering/Codes/data/iris.csv")
df.index.name = 'record'
# define columns to filter
cols = ['sepal length in cm',
        'sepal width in cm',
        'petal length in cm',
        'petal width in cm',]
# We will use the support vector machine classifier (SVC) as the estimator 
# for our example RFE. Now, let's import our modules and define 
# the independent (X) and dependent (y) variables for the SVC object:
# load modules for RFE and the classifier SVC
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
# set independent vars to X and dependent var to y
X = df[cols]
y = df['species']
# Next, we will instantiate both the RFE and SVC object, and pass the SVC object 
# as an argument into RFE. We will use the n_features_to_select arg to choose 
# the number of output features (2, in this case). Then, we fit and check 
# the feature rankings with RFE's ranking_ attribute, as follows:
# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=2, step=1)
rfe.fit(X, y)

# print rankings
print(cols)
print(rfe.ranking_)

# Transformation
# This strategy chooses a few new dimensions, or feature vectors, to project
# the original data into. There are two common mathematical methods, both of which 
# are fully deterministic and targeted at either the supervised or unsupervised case.

# instantiate pca object with 2 output dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
# fit and transform using 2 input dimensions
out_pca = pca.fit_transform(df[['petal length in cm',
                                'petal width in cm',]])
# create pca output dataframe and add label column "species" 
df_pca = pd.DataFrame(data = out_pca, columns = ['pca1', 'pca2'])
df_pca = pd.concat([df_pca, df[['species']]], axis = 1)
# plot scatter of pca data
sns.lmplot(x='pca1', y='pca2', hue='species', data=df_pca, fit_reg=False)
# get variance explained by each component
print(pca.explained_variance_ratio_)









