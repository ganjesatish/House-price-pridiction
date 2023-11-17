#!/usr/bin/env python
# coding: utf-8

# In[1]:


path = 'C:\\Users\\SATISH\\Downloads\\'


# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[4]:


df = pd.read_csv(path+"data.csv")


# In[5]:


print(df.head())


# In[6]:


X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']]
y = df['price']


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']


# In[9]:


X_train.columns = feature_names
X_test.columns = feature_names


# In[10]:


model = LinearRegression()


# In[11]:


model.fit(X_train, y_train)


# In[12]:


y_pred = model.predict(X_test)


# In[13]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[14]:


print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[15]:


new_data = [[3, 2, 1500, 4000, 1, 0, 0, 3]]
predicted_price = model.predict(new_data)
print("Predicted Price:", predicted_price[0])


# In[ ]:




