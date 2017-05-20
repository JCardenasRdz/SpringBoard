
# coding: utf-8

# In[1]:

# import custom modules wrote by julio
import importlib
import seaborn as sns
from capstone_01 import clean_data
from capstone_01 import inferential_statistics

# reload modules without restartign the kernall (it drives me crazy)
importlib.reload(clean_data);
importlib.reload(inferential_statistics);


# In[2]:

# load data and cleant it
file = './raw_data/clinical/I-SPY_1_All_Patient_Clinical_and_Outcome_Data.xlsx'
df = clean_data.clean_my_data(file)
df.head(2)


# ## Inferential_statistics
# ### 1. Categorical vs Categorical

# In[3]:

# assign all categorical predictors
predictors = df.columns[1:7]
predictors


# In[4]:

# example of contingency table
inferential_statistics.contingency_table('ER+', 'PCR',df)


# In[5]:

inferential_statistics.categorical_data('PCR', predictors, df)


# In[6]:

# example of contingency table
inferential_statistics.contingency_table('White', 'Alive',df)


# In[7]:

# run chi-squared test on Survival as outcome
inferential_statistics.categorical_data('Alive', predictors, df)


# ## 2. Continous vs Ca## 2. Continous vs Categorical (ANOVA)
# ### 2.1 Effect of Age on PCA and Survivaltegorical (ANOVA)
# ### 2.1 Effect of Age on PCA and Survival

# In[8]:

predictor= ['age']
outcome = 'PCR'
anova_table, OLS = inferential_statistics.linear_models(df, outcome, predictor);
sns.boxplot(x= outcome, y=predictor[0], data=df, palette="Set3");


# In[9]:

predictor= ['age']
outcome = 'Alive'
anova_table, OLS = inferential_statistics.linear_models(df, outcome, predictor);
sns.boxplot(x= outcome, y=predictor[0], data=df, palette="Set3");


# ### explore interactions between age, survival, and PCR

# In[10]:

sns.boxplot(x= 'PCR', y='age', hue ='Alive',data=df, palette="Set3");


# ## The survival of patients that achieved PCR is affected by their age

# In[11]:

predictor= ['age']
outcome = 'Alive'
anova_table, OLS = inferential_statistics.linear_models(df.loc[df.PCR=='Yes',:], outcome, predictor);
sns.boxplot(x= outcome, y=predictor[0], data=df.loc[df.PCR=='Yes',:], palette="Set3");


# In[12]:

sns.boxplot(x= 'PCR', y='Survival_length', hue ='Alive',data=df, palette="Set3");


# ### 3.1 Effect of MRI measurements on PCR

# In[13]:

R = inferential_statistics.anova_MRI('PCR', df);


# ### 3.1 Effect of MRI measurements on Survival

# In[14]:

R = inferential_statistics.anova_MRI('Alive', df);


# In[ ]:
