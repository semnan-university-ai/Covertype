#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##### Author : Amir Shokri
##### github link : https://github.com/amirshnll/Covertype
##### dataset link : http://archive.ics.uci.edu/ml/datasets/Covertype
##### email : amirsh.nll@gmail.com


# In[1]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv('covtype_data.csv', header=None)


# In[3]:


df


# In[4]:


df.describe()


# In[13]:


x = df[df.columns[:54]]
y = df[df.columns[54]]
scaler = MinMaxScaler()
scaled_x = scaler.fit_transform(x)


# In[14]:


y.value_counts().plot.pie()


# In[6]:


#Dimentionality reduction
pca = PCA(n_components=15)
reduced_x = pca.fit_transform(scaled_x)


# In[7]:


#Choose whether reduces or not
X = scaled_x
X = reduced_x


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[10]:


#Now we run algorithms and evaluate


# In[11]:


from sklearn.naive_bayes import CategoricalNB
cnb = CategoricalNB()
cnb.fit(X_train, y_train)
predicted = cnb.predict(X_test)

print('MSE:', MSE(y_test, predicted))
print(classification_report(y_test, predicted))


# In[10]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', alpha=0.0001)
mlp.fit(X_train, y_train)
predicted = mlp.predict(X_test)

print('MSE:', MSE(y_test, predicted))
print(classification_report(y_test, predicted))


# In[15]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)

print('MSE:', MSE(y_test, predicted))
print(classification_report(y_test, predicted))


# In[13]:


from sklearn.tree import DecisionTreeClassifier
Dtree = DecisionTreeClassifier()
Dtree.fit(X_train, y_train)
predicted = Dtree.predict(X_test)

print('MSE:', MSE(y_test, predicted))
print(classification_report(y_test, predicted))


# In[14]:


from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression()
lreg.fit(X_train, y_train)
predicted = lreg.predict(X_test)

print('MSE:', MSE(y_test, predicted))
print(classification_report(y_test, predicted))


# In[ ]:




