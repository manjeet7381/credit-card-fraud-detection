#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Necessary Libraries for the uses of this project
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


#Import Credit Card dataset
Credit_df=pd.read_csv("creditcard.csv")


# In[3]:


#first five rows of the datafarame 
#it describe the dataframe
Credit_df.head() 


# In[4]:


#last 5 rows of the dateframe
Credit_df.tail()


# In[5]:


# this give the informtion about the credit card dataframe in terms of data
Credit_df.info()


# In[6]:


#checking if there is any missing values in the dataset
Credit_df.isnull().sum()


# In[7]:


Credit_df["Class"].value_counts()
# 0------Normal Transaction
# 1------Fradulent Transaction


# In[8]:


#seperating the data for analysis
legit = Credit_df[Credit_df["Class"] == 0]
fraud = Credit_df[Credit_df["Class"] == 1]


# In[9]:


print(legit.shape)
print(fraud.shape)


# In[10]:


#statistical measures of the data "Amount" in legit
legit["Amount"].describe()


# In[11]:


#Statistical measures of the "Amount" in 
fraud["Amount"].describe()


# In[12]:


#compare the values for both transactions grouping based on Class column
Credit_df.groupby("Class").mean()


# In[13]:


legit_sample = legit.sample(n = 492)


# In[14]:


new_dataset = pd.concat([legit_sample,fraud],axis=0)


# In[15]:


new_dataset.head()


# In[16]:


new_dataset.tail()


# In[17]:


new_dataset["Class"].value_counts()


# In[18]:


new_dataset.groupby("Class").mean()


# In[19]:


X = new_dataset.drop(columns = "Class",axis = 1)
Y = new_dataset["Class"]


# In[20]:


print(X)


# In[21]:


print(Y)


# In[22]:


X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape,X_train.shape,X_test.shape)


# In[23]:


model = LogisticRegression()


# In[24]:


#training the logistic regression model with training Data
model.fit(X_train,Y_train)


# In[25]:


#accuracy on training data
X_train_prediction= model.predict(X_train)
training_data_accuracy= accuracy_score(X_train_prediction,Y_train)


# In[26]:


print("Accuracy on Training data :",training_data_accuracy)


# In[27]:


#accuracy on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[28]:


print("Accuracy on Test data :",test_data_accuracy)


# In[ ]:




