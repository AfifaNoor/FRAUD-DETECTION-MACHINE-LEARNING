#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# In[2]:


df=pd.read_csv('Fraud.csv')


# In[3]:


df


# In[4]:


# The 'isFraud' column is the dependent variable in a fraud detection model, 
# representing whether a transaction is fraudulent (1) or not (0). 
# It serves as the target variable that the model aims to predict based on other features and patterns in the dataset,
# making it the variable of interest in the analysis.


# # Data Understanding

# In[5]:


df.info()


# In[6]:


df['step'].unique()


# In[7]:


df.describe()


# In[8]:


df.corr()['isFraud']


# ## 1. Data cleaning including missing values, outliers and multi-collinearity

# In[9]:


df.isnull().sum()


# In[10]:


# There is no missing and Null value in the Data.


# ## Is there any Duplicate Value?

# In[11]:


df.duplicated()


# In[12]:


#There is no duplicate value in the dataset


# # EDA

# In[13]:


ax=sns.countplot(df["type"])
ax.bar_label(ax.containers[0]) 


# In[14]:


#payment and cashout have faced more fraud 


# In[15]:


ax=sns.countplot(df["isFraud"])
ax.bar_label(ax.containers[0])


# In[16]:


plt.figure(figsize=(10, 5))
ax=sns.barplot(x=df['type'],y=df['isFraud'])
ax.set_title('Relashionship Between type and isFraud',fontdict={'size':15})


# In[17]:


#It appears that created a bar plot to visualize the relationship between transaction type 
#and the 'isFraud' column, and   'TRANSFER' transaction type has a high bar in the plot. 
#This suggests that 'TRANSFER' transactions might have a higher incidence of fraud compared to other transaction types.


# In[18]:


sns.heatmap(pd.crosstab(df["type"],df["isFraud"]))


# ## converting type variable in numerical 

# In[19]:


df['type'].unique()


# In[20]:


df['type'].value_counts()


# In[21]:


df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'],value=[2,4,1,5,3],inplace=True)


# In[22]:


df


# ### Drop Columns

# In[23]:


columns_to_drop = ['step', 'nameOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 
                   'isFlaggedFraud']
df.drop(columns=columns_to_drop,axis=1, inplace=True)


# In[24]:


df


# # Feature Engineering

# In[25]:


from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
df[["amount"]]=oe.fit_transform(df[["amount"]])
df[["oldbalanceOrg"]]=oe.fit_transform(df[["oldbalanceOrg"]])
df[["newbalanceOrig"]]=oe.fit_transform(df[["newbalanceOrig"]])


# In[26]:


df


# # Data Splitting

# In[27]:


x=df.iloc[:,0:-1].values
y=df.iloc[:,-1].values


# In[28]:


x


# In[29]:


y


# In[30]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[31]:


x_train


# In[32]:


x_test


# In[33]:


y_train


# In[34]:


y_test


# # Model Train

# In[35]:


from sklearn.ensemble import RandomForestClassifier


# In[36]:


rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)


# ## Prediction

# In[37]:


y_pred=rfc.predict(x_test)
y_pred


# # Model Evaluation

# In[38]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report
print("Classification report:",classification_report(y_pred,y_test))


# In[39]:


print("Accuracy:",accuracy_score(y_pred,y_test))


# In[ ]:





# In[40]:


"""1.	Data cleaning including missing values, outliers, and multi-collinearity.
Ans Data Cleaning - Replacing Categories with Numbers:
    In this step, I have replaced transaction types ('PAYMENT', 'TRANSFER', etc.) with numerical codes (2, 4, 1, 5, 3). This helps the model work with categorical data more effectively.
     Data Cleaning - Dropping Unnecessary Columns
  I have removed certain columns ('step', 'nameOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud') from the dataset. These columns are not needed for our analysis and can be dropped to simplify the data and reduce noise.

2. Describe your fraud detection model in elaboration
I have done many thing with this dataset,Firstly I have cleaned the data as there is no missing and duplicate value .
I have dropped some column and replaced ‘type’ column with a numerical number for better accuracy.

EDA- Exploratory Data Analysis 
I have done EDA on type and isFraud column , and found that transfer transaction is mostly use for farud .

Data Splitting 
As in this dataset the target variable  is ‘isFraud’ as it will help us to find either farud is happening and
how much is happening

Train model 
I have trained my model using RandomForest Classifier 

Model Evaluation 
I have used classification report to mae a report on accuracy,precision 


3.How did you select variables to be included in the model?
 The 'isFraud' column is the dependent variable in a fraud detection model, representing whether a transaction 
 is fraudulent (1) or not (0). It serves as the target variable that the model aims to predict based on other features 
 and patterns in the dataset,making it the variable of interest in the analysis.
 
 
4. Demonstrate the performance of the model by using best set of tools.
Classification report:               precision    recall  f1-score   support

           0       1.00      1.00      1.00   1270945
           1       0.88      0.91      0.89      1579

    accuracy                           1.00   1272524
   macro avg       0.94      0.96      0.95   1272524
weighted avg       1.00      1.00      1.00   1272524

5. What are the key factors that predict fraudulent customers?
 Unusual transaction behavior (frequency and amounts).
Geographic anomalies in transactions.
Suspicious account activity (creation, login patterns, balances).
Specific transaction types (e.g., 'TRANSFER' or 'CASH_OUT'). Deviations from historical behavior.
 Account-to-account interactions (unusual transfers or withdrawals).

6.Do these factors make sense? If yes, How? If not, How not?
Yes, these factors make sense for fraud prediction because they capture common characteristics and behaviors associated
with fraudulent activities, helping the model identify suspicious transactions effectively.

7.What kind of prevention should be adopted while company update its infrastructure?
Prevention against fraud during infrastructure updates is important for any organization.
By following some of the steps like security measures, staying vigilant, and educating both employees and users,
a company can reduce the risk of fraudulent activities.
It's an ongoing process that requires adapting to evolving threats and maintaining a strong culture of security awarenes.


8. Assuming these actions have been implemented, how would you determine if they work?
We need to check on a continuous and collect data and again implement machine learning to know
 how much it reduces and how much more need to work ."""





# In[ ]:





# In[ ]:




