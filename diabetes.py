
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import tree

data = pd.read_csv('https://raw.githubusercontent.com/mdyaqub14/TestSalesforceDX/0ec70f3bcb61efee3c9dd4eb951bdb78c427ffb6/Train.csv', index_col=0)
data.head()
data.info


# In[2]:


data['BMI'] = data['BMI'].astype(int)
data['DiabetesPedigreeFunction'] = data['DiabetesPedigreeFunction'].astype(int)


# In[3]:


features = list(data.columns[:8])
features


# In[4]:


y = data['Outcome']
x = data[features]
Tree = tree.DecisionTreeClassifier()
Tree = Tree.fit(x,y)


import graphviz

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = tree.export_graphviz(Tree, out_file='tree.dot',  
                         filled=True, rounded=True,  
                         special_characters=True) 


# In[5]:


output = Tree.predict([[10,115,0,0,0,0,0.261,30]])
print (output)


# In[6]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(x,y)


# In[7]:


output =  forest.predict([[4,118,70,0,0,44.5,0.904,26]])
print (output)

