from sklearn.datasets import load_iris
import scipy
iris = load_iris()
X = iris.data
Y = iris.target

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X,Y)
logreg.predict(X)

y_pred = logreg.predict(X)
len(y_pred)

from sklearn import metrics
print metrics.accuracy_score(Y, y_pred)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,Y)
y_pred = knn.predict(X)
print metrics.accuracy_score(Y, y_pred)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,Y)
y_pred = knn.predict(X)
print metrics.accuracy_score(Y, y_pred)

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state=4)
print x_train.shape
print x_test.shape

print y_train.shape
print y_test.shape

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test) 
print metrics.accuracy_score(y_test, y_pred)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print metrics.accuracy_score(y_test, y_pred)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print metrics.accuracy_score(y_test, y_pred)

k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

print scores

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.plot(k_range, scores)
plt.xlabel('value for k for knn')
plt.ylabel('testing accuracy')

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X,Y)
knn.predict([[3,5,4,2]])

import pandas as pd

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
data.head()
data.tail()
data.shape

import seaborn as sn
get_ipython().magic(u'matplotlib inline')
sn.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=7, aspect=0.7, kind='reg')

