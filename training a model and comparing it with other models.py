
# coding: utf-8

# In[4]:


from sklearn.datasets import load_iris
import scipy
iris = load_iris()
X = iris.data
Y = iris.target



# In[16]:





# In[17]:





# In[18]:





# In[23]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X,Y)
logreg.predict(X)


# In[22]:


y_pred = logreg.predict(X)
len(y_pred)


# In[28]:


from sklearn import metrics
print metrics.accuracy_score(Y, y_pred)


# In[30]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,Y)
y_pred = knn.predict(X)
print metrics.accuracy_score(Y, y_pred)


# In[31]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,Y)
y_pred = knn.predict(X)
print metrics.accuracy_score(Y, y_pred)


# In[35]:


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state=4)


# In[36]:


print x_train.shape
print x_test.shape


# In[37]:


print y_train.shape
print y_test.shape


# In[38]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)


# In[39]:


y_pred = logreg.predict(x_test) 
print metrics.accuracy_score(y_test, y_pred)


# In[40]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print metrics.accuracy_score(y_test, y_pred)


# In[41]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print metrics.accuracy_score(y_test, y_pred)


# In[42]:


k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))


# In[43]:


print scores


# In[44]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.plot(k_range, scores)
plt.xlabel('value for k for knn')
plt.ylabel('testing accuracy')


# In[45]:


knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X,Y)
knn.predict([[3,5,4,2]])


# In[46]:


import pandas as pd


# In[51]:


data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
data.head()


# In[52]:


data.tail()


# In[53]:


data.shape


# In[55]:


import seaborn as sn


# In[56]:


get_ipython().magic(u'matplotlib inline')


# In[62]:


sn.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=7, aspect=0.7, kind='reg')

