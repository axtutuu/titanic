import pandas as pd
import numpy as np
from sklearn import tree

pd.options.mode.chained_assignment = None

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

train.head()
test.head()

# print(train.shape)
# print(test.shape)

train.describe()
test.describe()


#############  データの前処理   ##################
def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * null_val / len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(columns = { 0 : '欠損数', 1 : '%' })
    return kesson_table_ren_columns


# kesson_table(train)
# kesson_table(test)



train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

# kesson_table(train)


train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# train.head(10)


test["Age"] = test["Age"].fillna(test["Age"].median())
# train["Embarked"] = train["Embarked"].fillna("S")

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

# test.head(10)


test.Fare[152] = test.Fare.median()

#############  データの前処理   ##################


#############  決定木   ##################

target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
my_prediction = my_tree_one.predict(test_features)

my_prediction.shape

#############  決定木   ##################
