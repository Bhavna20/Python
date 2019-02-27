#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Quantile regression to determine the impact of mother characteristics and smoking practice on child birth weight 

import numpy as np
import pandas as pd
import patsy
from scipy.optimize import minimize
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.regression.quantile_regression import QuantReg


# In[5]:


#read data
data=pd.read_csv("birthweight_smoking.csv")


# In[6]:


# At quantile 0.5
mod = smf.quantreg('birthweight ~ smoker + unmarried + educ + age + drinks + nprevist + alcohol + tripre1 + tripre2 + tripre3', data)
res = mod.fit(q=.5)
print(res.summary())


# In[7]:


#At three diffrent quantiles
quantiles = np.arange(.25, 1, .25)
def fit_model(q):
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['smoker'], res.params['unmarried'], res.params['educ'], res.params['age'], res.params['drinks'], res.params['nprevist'], res.params['alcohol'], res.params['tripre1'], res.params['tripre2'], res.params['tripre3']] 
mdl = [fit_model(x) for x in quantiles]
mdl = pd.DataFrame(mdl, columns=['q', 'a', 'b','c','d','e','f','g','h','i','j','k'])

ols= smf.ols('birthweight ~ smoker + unmarried + educ + age + drinks + nprevist + alcohol + tripre1 + tripre2 + tripre3', data).fit()
ols = dict(a = ols.params['Intercept'],
           b = ols.params['smoker'],
           c = ols.params['unmarried'],
           d = ols.params['educ'],
           e = ols.params['age'],
           f = ols.params['drinks'],
           g = ols.params['nprevist'],
           h = ols.params['alcohol'],
           i = ols.params['tripre1'],
           j = ols.params['tripre2'],
           k = ols.params['tripre3'])
print(mdl)
print(ols)


# In[8]:


get_ipython().system('jupyter nbconvert --to=python quantile_regression.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:




