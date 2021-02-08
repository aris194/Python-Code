#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Data Science Project - Yelp Reviews

import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import json

#cant import geopandas
#import geopandas as gpd

from matplotlib import style
style.use('dark_background')
print(pd.__version__)


# In[3]:


df_bus = pd.read_json('yelp_academic_dataset_business.json', lines=True , orient='columns')


# In[ ]:





# In[4]:


df_rev = pd.read_json('yelp_academic_dataset_review.json', lines=True , orient='columns')


# In[4]:





# In[5]:


len(df_rev)


# In[6]:


#df_check = pd.read_json('yelp_academic_dataset_checkin.json', lines=True , orient='columns')
#df_check.head()


# In[7]:


#df_tip = pd.read_json('yelp_academic_dataset_tip.json', lines=True , orient='columns')
#df_tip.head()


# In[8]:


df_user = pd.read_json('yelp_academic_dataset_user.json', lines=True , orient='columns')
df_user.head()


# In[9]:


#Insert map and geopandas
#bus_map.head()
#bus_map.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4)
#bus_map.plot (kind= 'scatter')
#plt.show
#import descartes 
#from mpl_toolkits.basemap import Basemap
#from shapely.geometry import Point, Polygon
#bus_map=df_bus[["longitude","latitude"]]

#latitude_list = [df_bus["latitude"]] 
#longitude_list = [df_bus["longitude"]] 
len(df_user)


# In[10]:


#Firstly I want to filter for metropolitan area the business.json file
bus_short=df_bus[["categories","business_id","city","state","review_count"]] #taking 3 columns from the business.json dataframe
rev_short=df_rev[["business_id","user_id"]] #taking 3 columns from the review.json dataframe
user_short=df_user[["user_id","review_count"]]


# In[ ]:





# In[148]:





# In[273]:


#filter specific rows from a DataFrame 
lv=bus_short[bus_short["city"] == 'Las Vegas']
#Now I want to find the empirical PDF of the number of reviews received for the individual business in LA
lv.head()


# In[86]:


##################### converting the dataframe to an array to find user IDs for RQ_1_1
import numpy as np
lvtoarr = lv["business_id"].to_numpy()
np.shape(lvtoarr)


# In[87]:


del df_bus #for memory purposes


# In[94]:


################This loop sorts by business id from the filtered business.json file and matches 
#it to the review file (similar to inner join merging for business_id) 
DF = []
for i in lvtoarr:
    df_loop = rev_short.loc[rev_short['business_id']==i]
    DF.append(df_loop)
df4 = pd.concat(DF)


# In[104]:


#######This sorts the array by user ids and counts

vals = df4.groupby('user_id').size().to_numpy()
vals.shape


# In[221]:


vals


# In[223]:


############################################################################################################
##########RQ.1_Subquestion_2 Plotting the empirical PDF & CDF in linear log for the business reviews##################
###########################################################################################################
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(context='paper', style='darkgrid', rc={'figure.facecolor':'white'}, font_scale=1.2)
fig, ax = plt.subplots()
# Plotting the empirical PDF in log log for the business reviews
xx=(lv['review_count'])

#normalize the empirical
#x = x/len(lv['review_count'])
left_lim = min(xx)
right_lim = max(xx)
 
plt.hist(xx, facecolor = 'peru', edgecolor = 'white', bins=200)
plt.xlabel('Number of reviews per business id', fontsize = 15, color= "black")
plt.title('Empirical PDF linear-log histogram of reviews per business in Las Vegas Metropolitan Area', fontsize =25, color ="black")
plt.yscale("log")
#plt.xscale("log")
plt.xlim(left_lim,right_lim)

#plt.ylabel('Important var')

plt.show()
print ('mean', xx.mean(), 'standard variation',xx.std())


# In[225]:


# Plotting the empirical CDF in log log for the business reviews
plt.hist(xx, facecolor = 'peru', edgecolor = 'white', bins=200, cumulative= True)
#plt.xlabel('Number of reviews per business id', fontsize = 15, color= "black")
plt.title('Empirical CDF log-log histogram of reviews per business in Las Vegas Metropolitan Area', fontsize =25, color ="black")
plt.yscale("log")
plt.xscale("log")
plt.xlabel('Number of reviews per business id', fontsize = 15, color= "black")
plt.show()


# In[275]:


#fitting the best continuous distribution using Kolmogorov test for evaluation
#Distribution fit for business reviews 
#https://medium.com/@amirarsalan.rajabi/distribution-fitting-with-python-scipy-bb70a42c0aed
plt.hist(xx, facecolor = 'peru', edgecolor = 'white', bins=200)
plt.xlabel('Number of reviews per business id', fontsize = 15, color= "black")
plt.title('Fitting multiple continuous PDFs in linear-log histogram of reviews per business in Las Vegas Metropolitan Area', fontsize =25, color ="black")
#plt.yscale("log")
size = 15000
x = scipy.arange(size)
results = []
for i in list_of_dists:
    dist = getattr (scipy.stats, i)
    param = dist.fit(lv['review_count'])
    a = scipy.stats.kstest(lv['review_count'], i, args=param)
    results.append((i,a[0],a[1]))
    pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size
    plt.plot(pdf_fitted, label=i)


plt.show()

results.sort(key=lambda x:float(x[2]), reverse=True)
for j in results:
    print("{}: statistic={}, pvalue={}".format(j[0], j[1], j[2]))
#The solution turns out to be the non central t distribution of NCT
#https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python


# In[224]:


############################################################################################################
#############RQ.1_Subquestion_1 Plotting the empirical CDF & PDF in linear log for the business reviews##################
###########################################################################################################

xl=vals
left_lim = min(xl)
right_lim = max(xl)

plt.hist(xl, facecolor = 'black', edgecolor = 'white', bins=200)
plt.xlabel('Number of reviews per business id', fontsize = 15, color= "black")
plt.title('Empirical PDF linear-log histogram of reviews per user in Las Vegas Metropolitan Area', fontsize =25, color ="black")
plt.yscale("log")
#plt.xscale("log")

#plt.ylabel('Important var')
plt.xlim(left_lim,right_lim)
plt.show()
print ('mean', xl.mean(), 'standard variation',xl.std())


# In[226]:


# Plotting the empirical CDF in log log for the user reviews
#xl=(new['review_count'])

#normalize the empirical
#xl = xl/len(new['review_count'])
left_lim = min(xl)
right_lim = max(xl)


plt.hist(xl, facecolor = 'black', edgecolor = 'white', bins=200, cumulative= True)
plt.xlabel('Number of reviews per business id', fontsize = 15, color= "black")
plt.title('Empirical PDF log-linear histogram of reviews per user in Las Vegas Metropolitan Area', fontsize =25, color ="black")
plt.yscale("log")
plt.xscale("log")

#plt.ylabel('Important var')

plt.show()


# In[254]:


############################################################################################################
####Alternative way to locate user ids by business_ids review counts using pandas for RQ1_Subquestion_1##################################
###########################################################################################################
################This sorts by user id 
lv=lv["business_id"]
merged_inner = pd.merge(left = user_short, right=rev_short, left_on='user_id', right_on='user_id', how = 'inner')
#merged_inner = merged_inner[["user_id", "business_id"]]
merged_inner


# In[256]:


################This sorts by user id 
merged_inner2 = pd.merge(left=merged_inner, right=lv, left_on='business_id', right_on='business_id', how = 'inner')
#new2 = pd.merge(new, user_short, on = "user_id")
#pd.options.display.max_rows = None
merged_inner2
#new2.head()


# In[262]:


############################################################################################################
##########RQ.1_Subquestion_1 Plotting the empirical PDF in linear log for the business reviews##################
###########################################################################################################

sns.set(context='paper', style='darkgrid', rc={'figure.facecolor':'white'}, font_scale=1.2)
fig, ax = plt.subplots()

xx=(merged_inner2['review_count'])

#normalize the empirical
#x = x/len(lv['review_count'])

 
plt.hist(xx, facecolor = 'peru', edgecolor = 'white', bins=200)
plt.xlabel('Number of reviews per business id', fontsize = 15, color= "black")
plt.title('Empirical PDF linear-log histogram of reviews/user in Las Vegas Metropolitan Area (Pandas)', fontsize =25, color ="black")
plt.yscale("log")
#plt.xscale("log")
#plt.xlim(left_lim,right_lim)

#plt.ylabel('Important var')

plt.show()
print ('mean', xx.mean(), 'standard variation',xx.std())


# In[269]:


sns.set(context='paper', style='darkgrid', rc={'figure.facecolor':'white'}, font_scale=1.2)
fig, ax = plt.subplots()

xx=(merged_inner2['review_count'])

#normalize the empirical
#x = x/len(lv['review_count'])

plt.hist(xx, facecolor = 'peru', edgecolor = 'white', bins=200, cumulative=True)
plt.xlabel('Number of reviews per business id', fontsize = 15, color= "black")
plt.title('Empirical CDF linear-log histogram of reviews/user in Las Vegas Metropolitan Area (Pandas)', fontsize =25, color ="black")
plt.yscale("log")


plt.show()


# In[270]:


import warnings
warnings.filterwarnings("ignore")
#list_of_dists = 'weibull_min','norm','weibull_max','beta','invgauss','uniform','gamma','expon','lognorm','pearson3','triang'
list_of_dists = 'alpha','anglit','arcsine','beta','betaprime','bradford','burr','burr12','cauchy','chi','chi2','cosine','dgamma','dweibull','erlang','expon','exponnorm','exponweib','exponpow','f','fatiguelife','fisk','foldcauchy','foldnorm','genlogistic','genpareto','gennorm','genexpon','genextreme','gausshyper','gamma','gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r','gumbel_l','halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant','invgamma','invgauss','invweibull','johnsonsb','johnsonsu','kstwobign','laplace','levy','levy_l','logistic','loggamma','loglaplace','lognorm','lomax','maxwell','mielke','nakagami','ncx2','ncf','nct','norm','pareto','pearson3','powerlaw','powerlognorm','powernorm','rdist','reciprocal','rayleigh','rice','recipinvgauss','semicircular','t','triang','truncexpon','truncnorm','tukeylambda','uniform','vonmises','vonmises_line','wald','weibull_min','weibull_max'
print(type(list_of_dists))


# In[278]:


#fitting the best continuous distribution using Kolmogorov test for evaluation
#Distribution fit for user reviews 
#https://medium.com/@amirarsalan.rajabi/distribution-fitting-with-python-scipy-bb70a42c0aed
results = []
for i in list_of_dists:
    dist = getattr (scipy.stats, i)
    param = dist.fit(vals)
    a = scipy.stats.kstest(vals, i, args=param)
    results.append((i,a[0],a[1]))
    
plt.show()

results.sort(key=lambda x:float(x[2]), reverse=True)
for j in results:
    print("{}: statistic={}, pvalue={}".format(j[0], j[1], j[2]))
#The solution turns out to be the non central t distribution of NCF
#ncf: statistic=0.30714449067693433, pvalue=0.0
#https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python


# In[202]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



          

