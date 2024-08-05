#!/usr/bin/env python
# coding: utf-8

# #   ENHANCING WEATHER FORECASTING ACCURACY

# In[ ]:





# In[ ]:





# # LOADING LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# # LOADING DATASET

# In[2]:


data=pd.read_csv("C:/pdata/DailyDelhiClimateTrain.csv")


# # ANALYZING DATA

# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


data.info()


# In[15]:


data.shape


# In[16]:


data.columns


# In[17]:


data.dtypes


# In[18]:


figure = px.line(data, x="date", 
                 y="meantemp", 
                 title='Mean Temperature in Delhi Over the Years')
figure.show()


# In[19]:


figure = px.line(data, x="date", 
                 y="humidity", 
                 title='Humidity in Delhi Over the Years')
figure.show()


# In[20]:


figure = px.line(data, x="date", 
                 y="wind_speed", 
                 title='Wind Speed in Delhi Over the Years')
figure.show()


# In[21]:


figure = px.scatter(data_frame = data, x="humidity",
                    y="meantemp", size="meantemp", 
                    trendline="ols", 
                    title = "Relationship Between Temperature and Humidity")
figure.show()


# In[22]:


data["date"]=pd.to_datetime(data["date"],format='%Y-%m-%d')
data['year']=data['date'].dt.year
data["month"]=data["date"].dt.month
data.head()


# In[23]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.title("Temperature Change in Delhi Over the Years")
sns.lineplot(data = data, x='month', y='meantemp', hue='year')
plt.show()


# In[24]:


forecast_data = data.rename(columns = {"date": "ds", 
                                       "meantemp": "y"})
print(forecast_data)


# # PREDICTION

# In[25]:


from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
model = Prophet()
model.fit(forecast_data)
forecasts = model.make_future_dataframe(periods=365)
predictions = model.predict(forecasts)
plot_plotly(model, predictions)


# In[ ]:




