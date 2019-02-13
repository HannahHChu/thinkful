import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy import stats
from sklearn import datasets, linear_model

df = pd.read_csv('winemag-data_first150k.csv')
print(df)

list(df)

gdf = df.dropna(how='any', axis='rows', subset=['price'], inplace=False)


# In[6]:
gdf.describe()


# In[7]:


##parsing out price and point column and grouped them by country

country_pp = gdf.groupby('country').aggregate(np.mean)
rating = country_pp['points']
cost = country_pp['price']
print(country_pp)

un_gdf=gdf['country'].unique()
print(un_gdf)

# In[8]:


##equation for line of best fit -- obtained from
###"https://stackoverflow.com/questions/22239691/code-for-line-of-best-fit-of-a-scatter-plot-in-python"

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b


# In[9]:


# solution
a, b = best_fit(cost, rating)
##best fit line:
##y = 82.88 + 0.14x


# In[10]:


##Scatter plot of price vs points
plt.scatter(
    x=country_pp['price'],
    y=country_pp['points'],
    color='red',
    marker='o', s=10
)
plt.ylabel('Points')
plt.xlabel('Prices (USD)')
plt.title('Average Cost of Wine vs. WineEnthusiast Ratings Based on Country')

#line of best fit
yfit = [a + b * xi for xi in cost]
plt.plot(cost, yfit, color='black', linewidth='0.25')
plt.show()

####To see whether the cost of wine is correlated with its taste
#####(assumed by a high WineEnthusiast rating)


##Prices
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(cost, color='green', bins=12)
plt.ylabel('Frequency')
plt.xlabel('Price (USD)')
plt.title('Most Common Wine Prices')

plt.subplot(1, 2, 2)
plt.boxplot(cost, vert=False)
plt.xlim(0,100)
plt.xlabel('Price (USD)')
plt.title('Most Common Wine Prices')
plt.show()

##Points
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(rating, color='green', bins=12)
plt.ylabel('Frequency')
plt.xlabel('WineEnthusiast Rating')
plt.title('WineEnthusiast Ratings')

plt.subplot(1, 2, 2)
plt.boxplot(rating, vert=False)
plt.xlabel('WineEnthusiast Rating')
plt.title('WineEnthusiast Ratings')
plt.show()





###Boxplot of all countries showing prices
##Example with the US -- try Loop
gdf2 = gdf.set_index('country', inplace=False)
us_cost=gdf2.loc[['US'],['price']]
us_summ = us_cost['price']
us_summ.describe()

plt.boxplot(us_summ)
plt.show()

##loop

def summ(country):
    for nam in country:
        print(nam)
        if str(nam) == str('nan'):
            continue
        cost=gdf2.loc[[nam],['price']]
        result=cost['price']
        plt.boxplot(cost)
        plt.show()

summ(un_gdf)
