#!/usr/bin/env python
# coding: utf-8

# ## Quiz

# In[1]:


from bs4 import BeautifulSoup
import os
import pandas as pd


# In[22]:


# List of dictionaries to build file by file and later convert to a DataFrame
df_list = []
folder = 'rt_html'
for movie_html in os.listdir(folder):
    with open(os.path.join(folder, movie_html)) as file:
        Soup=BeautifulSoup(file, 'lxml')
        #title = soup.find('title').contents[0][:len('- Rotten Tomatoes')]
        title=Soup.find('title').contents[0][:-len(' - Rotten Tomatoes')]
        audience_score=Soup.find('div',class_='audience-score meter').find('span').contents[0][:-1]
        num_of_audience_ratings = Soup.find('div', class_='audience-info hidden-xs superPageFontColor')
        num_of_audience_ratings = num_of_audience_ratings.find_all('div')[1].contents[2].strip().replace(',','')
    #print(title)
    #break
       
df_list.append({'title': title,'audience_score': int(audience_score),'number_of_audience_ratings': int(num_of_audience_ratings)})
#df = pd.DataFrame(df_list, columns = ['title', 'audience_score', 'number_of_audience_ratings'])
df = pd.DataFrame(df_list, columns = ['title', 'number_of_audience_ratings','audience_score'])


# ## Solution Test
# Run the cell below the see if your solution is correct. If an `AssertionError` is thrown, your solution is incorrect. If no error is thrown, your solution is correct.

# In[23]:


df_solution = pd.read_pickle('df_solution.pkl')
df.sort_values('title', inplace = True)
df.reset_index(inplace = True, drop = True)
df_solution.sort_values('title', inplace = True)
df_solution.reset_index(inplace = True, drop = True)
pd.testing.assert_frame_equal(df, df_solution)


# In[24]:


df.head()


# In[ ]:




