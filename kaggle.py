#Get the Data https://www.kaggle.com/c/titanic/data

import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

full_train = pd.read_csv('train.csv')
kaggle_test = pd.read_csv('test.csv')
baseline_submission = pd.read_csv('gender_submission.csv')

full_train.describe(include='all')

full_train['Embarked'].value_counts()

fields = ['Survived', 'Age', 'Fare', 'Parch', 'SibSp', 'Pclass']
full_train[fields].hist(bins=25, figsize=(20,15))

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
splits = split.split(full_train, full_train['Pclass'])
train_indices, test_indices = next(splits)
train = full_train.iloc[train_indices]
test = full_train.iloc[test_indices]

original_train = train.copy()  # we're keeping a copy for later

survived = train[train['Survived'] == True].groupby('Pclass').size()
died = train[train['Survived'] == False].groupby('Pclass').size()

data = pd.concat([survived, died], axis=1)
data.columns = ['Survived', 'Died']
data.plot.bar(title='Outcome vs First, Second, or Third Class')

def f(age):  
    try:
        return int(age/10)*10
    except:
        pass
survived = train[train['Survived'] == True].copy().set_index('Age').groupby(f).size()
died = train[train['Survived'] == False].copy().set_index('Age').groupby(f).size()
data = pd.concat([survived, died], axis=1)
data.columns = ['Survived', 'Died']
data.plot.bar(title='Outcome vs Age')

survived = train[train['Survived'] == True].groupby('SibSp').size()
died = train[train['Survived'] == False].groupby('SibSp').size()

data = pd.concat([survived, died], axis=1)
data.columns = ['Survived', 'Died']
data.plot.bar(title='Outcome vs Number of Siblings or Spouses Aboard')

survived = train[train['Survived'] == True].groupby('Parch').size()
died = train[train['Survived'] == False].groupby('Parch').size()

data = pd.concat([survived, died], axis=1)
data.columns = ['Survived', 'Died']
data.plot.bar(title='Outcome vs Number of Parents or Children Aboard')

survived = train[train['Survived'] == True].groupby('Sex').size()
died = train[train['Survived'] == False].groupby('Sex').size()

data = pd.concat([survived, died], axis=1)
data.columns = ['Survived', 'Died']
data.plot.bar(title='Outcome vs Sex')

survived = train[train['Survived'] == True].groupby('Embarked').size()
died = train[train['Survived'] == False].groupby('Embarked').size()

data = pd.concat([survived, died], axis=1)
data.columns = ['Survived', 'Died']
data.plot.bar(title='Outcome vs Place Where Embarked')


#Prepare the Data for Machine Learning Algorithms
#Get rid of fields we won't be using

train = train.drop(['Name', 'Ticket'], axis=1)

# deal with nulls
train.isnull().sum()

# If too few values in a field (like Cabin), then we might as well just get rid of the field.
train = train.drop('Cabin', axis=1)



#If there is just a tiny bit of missing data (like Embarked) then we can get rid of those rows.
train = train.dropna(subset=['Embarked'])

from sklearn.preprocessing import Imputer

# set asside non-numerical fields
non_numerical_fields = ['Sex', 'Embarked']
train_numerical = train.drop(non_numerical_fields, axis=1)

# create imputer
imputer = Imputer(strategy='median')

# calculate medians for each field
imputer.fit(train_numerical)  # imputer.statistics_ contains vector of medians

# fill in missing values with medians
train_numerical_imputed = imputer.transform(train_numerical)

# stick back into DataFrame
train_numerical_imputed = pd.DataFrame(
    train_numerical_imputed, 
    columns=train_numerical.columns,
)
train_numerical_imputed.isnull().sum()

from sklearn.preprocessing import LabelBinarizer

binarizer = LabelBinarizer()

binarizer.fit(train['Sex'])

train_sex = binarizer.transform(train['Sex'])

train_sex[:10]

binarizer.fit(train['Embarked'])
train_embarked = binarizer.transform(train['Embarked'])
train_embarked[:10]


#Feature scaling

#Scaling the fields to similar ranges of values makes typically improves the performance of machine learning algorithms.

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train_numerical_imputed)
scaler.transform(train_numerical_imputed)


# Transformation Pipelines

# Rather than run every cell of this notebook to get the data into shape, it would be nice to make this process repeatable.

# You can with "Transformation Pipelines".
# Transformers

# Common pattern in Scikit Learn: fit and transform.

#     Imputers fit the data to find the median and transform the data to fill in the missing data.
#     Scalers fit the data to find min and max values and transform the data to scale every thing into the range of 0 to 1.
#     Classifiers fit the data to a model and predict category given new data.

# You can make your own Transformers. This one takes in a DataFrame and pulls out the fields you need.

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Allows you to specify a DataFrame as the input to a Pipeline (see below)"""
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values


# Pipelines

# Transformers by themselves seem like overkill until you see how they can be arranged into Pipelines.

# First let's create a pipeline for our numerical fields.

from sklearn.pipeline import Pipeline

numeric_pipeline = Pipeline([
    ('selector', DataFrameSelector(['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])),
    ('imputer', Imputer(strategy='median')),
    ('scaler', MinMaxScaler()),
])

numeric_pipeline.fit(train)
numeric_pipeline.transform(train)

# or equivalently...
numeric_pipeline.fit_transform(train)

categorical_pipeline = Pipeline([
    ('selector', DataFrameSelector(['Sex','Embarked'])),
    ('binarizer', LabelBinarizer()),
])

categorical_pipeline.fit_transform(train)

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(
    transformer_list=[
        ('numeric_pipeline', numeric_pipeline),
        ('categorical_pipeline', categorical_pipeline),
    ]
)

full_pipeline.fit_transform(train)


# Select and Train a Model

# A LogisticRegression model maps a bunch of numerical inputs to a binary output. Seems like a good candidate model for our current problem.

from sklearn.linear_model import LogisticRegression

train_prepared = full_pipeline.fit_transform(train)
train_true_outcome = train['Survived']

log_reg_model = LogisticRegression()
log_reg_model.fit(train_prepared, train_true_outcome)

from sklearn.metrics import accuracy_score

predictions = log_reg_model.predict(train_prepared)

accuracy_score(
    y_true=train_true_outcome,
    y_pred=predictions,
)

test_prepared = full_pipeline.transform(test)
test_true_outcome = test['Survived']

predictions = log_reg_model.predict(test_prepared)

accuracy_score(
    y_true=test_true_outcome,
    y_pred=predictions,
)

# ....................(shortcut to end if the talk is running long).....................

# Can we do better with a DecisionTreeClassifier?

# Let's try something crazy. I'm going to extend to the previous data preparation by finding polynomial combinations of features and then fit a DecisionTreeM

from sklearn.preprocessing import PolynomialFeatures

extended_pipeline = Pipeline([
    ('full_pipeline', full_pipeline),
    ('polynomial', PolynomialFeatures(degree=5)),
])
train_extra_prepared = extended_pipeline.fit_transform(train)
train_extra_prepared.shape

from sklearn.tree import DecisionTreeClassifier

dec_tree_model = DecisionTreeClassifier(max_leaf_nodes=None, random_state=43)
dec_tree_model.fit(train_extra_prepared, train_true_outcome)

accuracy_score(
    y_true=train_true_outcome,
    y_pred=dec_tree_model.predict(train_extra_prepared),
)

test_extra_prepared = extended_pipeline.transform(test)

accuracy_score(
    y_true=test_true_outcome,
    y_pred=dec_tree_model.predict(test_extra_prepared),
)


# Better Evaluation Using Cross Validation

#     Cross validation chops up the training set into 3 chunks.
#     It then builds the model with 2 chunks and evaluates on a third chunk.
#     It does this 3 times.
#     This is "fair" because the each model is evaluated against data it has not yet seen.


from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    dec_tree_model,
    train_extra_prepared,
    train_true_outcome,
    scoring='accuracy',
)
print(scores)


# Fine-Tune Your Model¶

from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [2, 5, 10, 99999]
}
dec_tree_model = DecisionTreeClassifier()
grid_search = GridSearchCV(
    dec_tree_model,
    param_grid,
    scoring='accuracy',
)
grid_search.fit(train_prepared, train_true_outcome)
grid_search.best_params_

best_dec_tree_model = grid_search.best_estimator_

test_prepared = full_pipeline.transform(test)  # NOTE: we've gone back to a less crazy pipeline

accuracy_score(
    y_true=test_true_outcome,
    y_pred=best_dec_tree_model.predict(test_prepared),
)

