#!/usr/bin/env python
# coding: utf-8

# # Data Mining

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv('diabetes.csv')#, encoding='latin-1')


# In[88]:


data.head(10)


# In[89]:


data.columns


# In[90]:


data.shape


# In[91]:


data.info


# In[92]:


data.isna().any()


# In[93]:


data.isnull().sum()


# In[94]:


[features for features in data.columns if data[features].isnull().sum()>0]


# In[95]:


sns.heatmap(data.isnull(), yticklabels=True,cbar=True)


# In[96]:


data.describe()


# In[97]:


data['Glucose'].value_counts().sum()


# In[98]:


data['Glucose']=data['Glucose'].replace(0,np.nan)


# In[99]:


data['BloodPressure']=data['BloodPressure'].replace(0,np.nan)


# In[100]:


data['SkinThickness']=data['SkinThickness'].replace(0,np.nan)


# In[101]:


data['Insulin']=data['Insulin'].replace(0,np.nan)


# In[102]:


data['BMI']=data['BMI'].replace(0,np.nan)


# In[103]:


data['DiabetesPedigreeFunction']=data['DiabetesPedigreeFunction'].replace(0,np.nan)


# In[104]:


data['Age']=data['Age'].replace(0,np.nan)


# In[105]:


data.head()


# In[106]:


data.isna().any()


# In[107]:


data.columns


# In[108]:


data['BMI']=data['BMI'].replace(0,np.nan)


# In[109]:


data['Pregnancies']=data['Pregnancies'].replace(0,np.nan)


# In[110]:


data['Glucose']=data['Glucose'].replace(0,np.nan)


# In[111]:


data['BloodPressure']=data['BloodPressure'].replace(0,np.nan)


# In[112]:


data['SkinThickness']=data['SkinThickness'].replace(0,np.nan)


# In[113]:


data['Insulin']=data['Insulin'].replace(0,np.nan)


# In[114]:


data['DiabetesPedigreeFunction']=data['DiabetesPedigreeFunction'].replace(0,np.nan)


# In[115]:


data['Age']=data['Age'].replace(0,np.nan)


# In[116]:


data.isnull().sum()


# In[117]:


sns.heatmap(data.isnull(), yticklabels=True,cbar=True)


# In[118]:


data.describe()


# In[119]:


data.fillna(data.median(),inplace=True)


# In[120]:


data.isnull().sum()


# # outerlier detection and treatment and plot boxplot for detecting outerlier

# In[121]:


data.columns


# In[122]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,1)
sns.boxplot(data['Pregnancies'])


# In[123]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,2)
sns.boxplot(data['Glucose'])


# In[124]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,3)
sns.boxplot(data['BloodPressure'])


# In[125]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,4)
sns.boxplot(data['SkinThickness'])


# In[126]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,5)
sns.boxplot(data['Insulin'])


# In[127]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,6)
sns.boxplot(data['BMI'])


# In[128]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,7)
sns.boxplot(data['DiabetesPedigreeFunction'])


# In[129]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,8)
sns.boxplot(data['Age'])


# In[130]:


data.columns


# In[131]:


data['Pregnancies']=data['Pregnancies'].clip(lower=data['Pregnancies'].quantile(0.95))


# In[132]:


data['Glucose']=data['Glucose'].clip(lower=data['Glucose'].quantile(0.95))


# In[133]:


data['BloodPressure']=data['BloodPressure'].clip(lower=data['BloodPressure'].quantile(0.95))


# In[134]:


data['SkinThickness']=data['SkinThickness'].clip(lower=data['SkinThickness'].quantile(0.95))


# In[135]:


data['Insulin']=data['Insulin'].clip(lower=data['Insulin'].quantile(0.95))


# In[136]:


data['BMI']=data['BMI'].clip(lower=data['BMI'].quantile(0.95))


# In[137]:


data['DiabetesPedigreeFunction']=data['DiabetesPedigreeFunction'].clip(lower=data['DiabetesPedigreeFunction'].quantile(0.95))


# In[138]:


data['Age']=data['Age'].clip(lower=data['Age'].quantile(0.95))


# In[139]:


#data['Outcome']=data['Outcome'].clip(lower=data['Outcome'].quantile(0.95))


# # again apply boxplot

# In[140]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,1)
sns.boxplot(data['Pregnancies'])


# In[141]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,2)
sns.boxplot(data['Glucose'])


# In[142]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,3)
sns.boxplot(data['BloodPressure'])


# In[143]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,4)
sns.boxplot(data['SkinThickness'])


# In[144]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,5)
sns.boxplot(data['Insulin'])


# In[145]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,6)
sns.boxplot(data['BMI'])


# In[146]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,7)
sns.boxplot(data['DiabetesPedigreeFunction'])


# In[147]:


plt.figure(figsize=(20,15))
plt.subplot(4,4,8)
sns.boxplot(data['Age'])


# # Data Visualisation

# In[148]:


sns.countplot(data['Outcome'])


# In[156]:


f,ax=plt.subplots(figsize=(20,10))
corr=data.corr("pearson")
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),cmap=sns.diverging_palette(220,10,as_cmap=True), square= True, ax=ax, annot=True)


# In[157]:


sns.pairplot(data,hue='Outcome',diag_kind='kde')


# In[ ]:




