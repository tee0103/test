#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly.**
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[3]:


df=pd.read_csv('ab_data.csv')


# In[4]:


df.head(3)


# b. Use the cell below to find the number of rows in the dataset.

# In[5]:


df.shape


# c. The number of unique users in the dataset.

# In[6]:


df.user_id.nunique()


# d. The proportion of users converted.

# In[7]:


conv_user=df.query('converted=="1"')['user_id'].nunique()/df.user_id.nunique()
conv_user


# In[8]:


df.head(3)


# e. The number of times the `new_page` and `treatment` don't match.

# In[19]:


non_match=df.query('group=="treatment" & landing_page=="old_page" or group=="control" & landing_page=="new_page" or group=="control" & landing_page=="old_page"')['user_id'].count()
non_match


# In[18]:


#df.landing_page.unique()
##array(['control', 'treatment'], 
#array(['old_page', 'new_page']
#df.query('group=="control" & landing_page=="old_page"')['user_id'].count()


# f. Do any of the rows have missing values?

# In[20]:


df.info()


# -  No, there are zero missing values

# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[21]:


test=df.query('group=="treatment" & landing_page=="new_page"')
df2=test.append(df.query('group=="control" & landing_page=="old_page"'))
df2.head(3)


# In[22]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[23]:


df2.user_id.nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[24]:


#df2[df2.duplicated(keep=false)]
df2[df2.user_id.duplicated(keep=False)]


# 773192

# # c. What is the row information for the repeat **user_id**? 

# In[28]:


df2.query('user_id=="773192"').info()


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[29]:


#df2 = df2.sort_values('timestamp', ascending=False)
df2=df2.drop_duplicates(subset='user_id',keep='first', inplace=False)
df2.head(3)


# In[30]:


#df2[df2.user_id.duplicated(keep=False)]
df2.query('user_id=="773192"')
#2896


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[31]:


df2.shape[0]
pro_conv=df2.query('converted=="1"')['user_id'].count()/df2.shape[0]
pro_conv


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[32]:


cont_conv=df2.query('converted=="1" & group=="control"')['user_id'].count()/df2.shape[0]
cont_conv


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[33]:


treat_conv=df2.query('converted=="1" & group=="treatment"')['user_id'].count()/df2.shape[0]
treat_conv


# d. What is the probability that an individual received the new page?

# In[34]:


rec_new=df2.query('landing_page=="new_page"')['user_id'].count()/df2.shape[0]
rec_new


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# **Your answer goes here.**

# - Based on the results, there is not enough evidence to state that the new treatment page leads to more conversions. 50%
# of individuals received the new page, with a chance of 11% converting. but we do not know if there are any biased behaviour taking place. In order to understand these results more, we will have to perform sampling distrubution and bootsraping to delve in deeper. This will give us a better reading. 

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **Put your answer here.**
# - H0:Pnew <= Pold
# - H1:Pnew > Pold

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# In[36]:


samp_ab=df.sample(df.shape[0], replace=True)
samp_ab.head(3)


# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[37]:


conv_pnew=samp_ab.query('landing_page=="new_page" & converted==True')['user_id'].nunique()/samp_ab.user_id.nunique()
conv_pnew
#conv_pnew=samp_ab.query('landing_page=="new_page" & group=="treatment" & converted=="1"')['user_id'].nunique()/samp_ab.shape[0]
#conv_pnew


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[38]:


conv_pold=samp_ab.query('landing_page=="old_page" & converted==True')['user_id'].nunique()/samp_ab.user_id.nunique()
conv_pold


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[39]:


samp_ab.query('group=="treatment"')['user_id'].nunique()


# d. What is $n_{old}$, the number of individuals in the control group?

# In[40]:


samp_ab.query('group=="control"')['user_id'].nunique()


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[41]:


new_page_converted=samp_ab.query('landing_page=="new_page" & group=="treatment" & converted==True')['user_id'].nunique()/samp_ab.user_id.nunique()
#new_page_converted=samp_ab.query('landing_page=="new_page" & group=="treatment" & converted=="1"')['user_id'].nunique()/samp_ab.user_id.nunique()
new_page_converted


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[42]:


old_page_converted=samp_ab.query('landing_page=="old_page" & group=="control" & converted==True')['user_id'].nunique()/samp_ab.user_id.nunique()
old_page_converted


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[43]:


obs_diff=new_page_converted - old_page_converted
obs_diff


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[49]:


new_converted_simulation = np.random.binomial(samp_ab.query('group=="treatment"')['user_id'].nunique(), conv_pnew, 10000)/samp_ab.query('group=="treatment"')['user_id'].nunique()
old_converted_simulation = np.random.binomial(samp_ab.query('group=="control"')['user_id'].nunique(), conv_pold, 10000)/samp_ab.query('group=="control"')['user_id'].nunique()
p_diffs = new_converted_simulation - old_converted_simulation




   
p_diffs = np.array(p_diffs)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[50]:


plt.hist(p_diffs);
plt.axvline(x=obs_diff,color='r');


# - yes, the plot matches my expectation. by increasing the sampling size, 
# the distribution will tend to form a normal distribution, which means that the sample statistic value is close enough to matching 
# the parameter value.

# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[ ]:





# In[51]:


null_vals=np.random.normal(0,p_diffs.std(),p_diffs.size) 
(null_vals > obs_diff).mean()


# In[52]:


plt.hist(null_vals);
plt.axvline(x=obs_diff, color='r')


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **Put your answer here.**

# - The value above refers to the P-value. Based on the results, it looks like our observed statistic (obs_diff) came 
# from this Null distribution as it lies on the distribution, rather than it been far away from it.
# Therefore, I do not reject the H0 hypothesis.

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[53]:


import statsmodels.api as sm

convert_old = old_page_converted
convert_new = new_page_converted
n_old = samp_ab.query('landing_page=="old_page")['user_id'].nunique()
#'landing_page=="old_page"
#samp_ab.query('group=="control"')['user_id'].nunique()
n_new = samp_ab.query('landing_page=="new_page")['user_id'].nunique()
#samp_ab.query('group=="treatment"')['user_id'].nunique()


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[58]:


stats, pval=stats.proportions_ztest(10,1000,0)
print(stats)


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **Put your answer here.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Put your answer here.**

# - Simple Linear regression

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[59]:


df2.head(3)


# In[60]:


df2['intercept']=1


# In[61]:


df2[['ab_page','old_page']]=pd.get_dummies(df2['landing_page'])

df2.tail(10)


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# - based on the coef of the new page(ab_page), users are less likely to convert 
# as compared to the old page due to its negative value.

# In[62]:



lm = sm.OLS(df2['converted'], df2[['intercept', 'ab_page']])
results = lm.fit()
results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# **Put your answer here.**

# - p-value =0.190. This value differs compared to that part 2 as it now indicates that we reject the null value and accept the alternative.

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Put your answer here.**

# - The more variables we use in our model, it allows us to see if these variables are related in 
# increasing the likely hood of our hypotheseis. Adding too many variables can lead to cases where a liner regression fails to exist, outliers occur or even correclated errors.
# For example, multicollinearity can flip the behaviour of the x-variable and the response. individually, there might be several variables that may relate to one another but once we combine them, they have an opposite coefficient.
# 

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[63]:


df_con=pd.read_csv('countries.csv')
df_con.head(5)


# In[64]:


df3=df2.join(df_con, how='inner',lsuffix='user_id', rsuffix='user_id')
df3.head(5)


# In[65]:


#df2.country.unique()
country_dummies = pd.get_dummies(df3['country'])
df_new = df3.join(country_dummies)
df_new.head()


# In[66]:


lm2 = sm.OLS(df_new['converted'], df_new[['intercept', 'CA','UK']])
results2 = lm2.fit()
results2.summary()


# US users are more related to convertions although UK users are expected to have a better converstion rate and Canadians with the 
# third most converted.

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# 
# 
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# In[ ]:




