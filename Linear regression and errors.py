feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
X = data[['TV', 'radio', 'newspaper']]
X.head()
print type(X)
print X.shape
print X.shape


Y.shape
Y = data['sales']
Y = data.sales
Y.head()
print type(Y)
print Y.shape
<class 'pandas.core.series.Series'>
(200,)

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=1)
​

print y_test.shape
print x_train.shape
print x_test.shape
print y_train.shape
print y_test.shape
(150, 3)
(50, 3)
(150,)
(50,)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

print linreg.intercept_
print linreg.coef_
2.8769666223179353
[0.04656457 0.17915812 0.00345046]

inreg.coef_
zip(feature_cols, linreg.coef_)
[('TV', 0.04656456787415026),
 ('radio', 0.1791581224508884),
 ('newspaper', 0.0034504647111804065)]

y_pred = linreg.predict(x_test)
 

#Module evaluation metrics for regression
true = [100, 50, 30, 20]
pred = [90, 50, 50, 10]

, pred
#calculate mean absolute error by hand
print (10+0+20+10)/4
#calculate by scikit learn
from sklearn import metrics
print metrics.mean_absolute_error(true, pred)
​
10
10.0

#calculate mean squared error by hand
print (10**2 + 0**2 + 20**2 + 10**2)/4
#calculate by scikit learn
print metrics.mean_squared_error(true, pred)
150
150.0

mean
#calculate root mean squared error by hand
import numpy as np
print np.sqrt((10**2 + 0**2 + 20**2 + 10**2)/4)
#calculate by scikit learn
print np.sqrt(metrics.mean_squared_error(true, pred))
12.24744871391589
12.24744871391589

y_test, y_pred
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))
1.4046514230328948