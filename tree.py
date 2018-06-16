import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier

import sys
sys.path.append("..")

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
x_labels = ['Pclass', 'Fare', 'Age', 'Sex']
x = data.loc[:, x_labels]
y = data['Survived']
x['Sex'] = x['Sex'].map(lambda sex: 1 if sex == 'male' else 0)
x = x.dropna(subset=x_labels)
y = y[x.index.values]

clf = DecisionTreeClassifier(random_state=241)
clf.fit(np.array(x.values), np.array(y.values))

importances = pandas.Series(clf.feature_importances_, index=['Pclass', 'Fare', 'Age', 'Sex'])
print(importances)
