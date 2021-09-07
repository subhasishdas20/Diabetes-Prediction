#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


df=pd.read_csv(r"C:\Learning\python_class\Logistic\diabetes.csv")


# In[3]:


df


# In[5]:


df.head()


# In[7]:


df.shape


# In[12]:


df.dtypes


# # #Finding Missing Values

# In[14]:


df.isnull().sum()


# ## Classifying X & Y 

# In[15]:


x= df.iloc[:,0:7]


# In[16]:


x


# In[19]:


y=df.iloc[:,-1]


# In[20]:


y


# ##Dummies

# In[25]:


df=pd.get_dummies(df,columns=["Class"], drop_first= True)


# In[26]:


df


# In[27]:


y=df.iloc[:,-1]


# In[28]:


y


# ## Scaling X Feature

# In[30]:


sc = StandardScaler()
x_scale = sc.fit_transform(x)


# In[31]:


x_scale


# ## Splitting Dataset

# In[34]:


X_train,X_test,Y_train, Y_test= train_test_split(x_scale, y, test_size=0.2)


# In[35]:


knn= KNeighborsClassifier(5)
knn.fit(X_train,Y_train)


# In[36]:


knn_pred = knn.predict(X_test)


# In[37]:


knn_pred


# In[38]:


from sklearn.metrics import accuracy_score, mean_squared_error
AccuracyScore=accuracy_score(Y_test, knn_pred)
Error=mean_squared_error(Y_test,knn_pred )


# In[39]:


AccuracyScore


# In[40]:


Error


# In[44]:


for i in range(1,20):
    knn= KNeighborsClassifier(i)
    knn.fit(X_train,Y_train)
    knn_pred = knn.predict(X_test)
    AccuracyScore=accuracy_score(Y_test, knn_pred)
    print (i, AccuracyScore)


# In[ ]:




