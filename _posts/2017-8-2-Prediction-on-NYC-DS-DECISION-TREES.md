---
layout: post
title: Predicting Gender of the riders of New York’s Citi Bikes (with Decision Trees)
---

### Read more on how I achieved 99.64% of accuracy with Decision Trees algorithm and Basic Feature Engineering.

 I was in Manhattan, New York for few weeks this June and as any tourist would do, I decided to ride around the city. One can easily go around the city using the subway, buses & taxis, I decided to have some fun and ride a bicycle. Renting a bike for the entire day costs around $40-$50. However, I found the Citi Bikes (shared bicycle service run by Citi Group) to be a better option. A day’s pass was $12 + taxes (~$13 in total) more info about the pricing on [Citi bikes website](https://www.citibikenyc.com/pricing).

I must say the experience was wonderful and so was the weather on that day. The temperature was around 70 degrees Fahrenheit (or 21.1 Degree Celsius). Riding a bicycle is one of the cheapest ways to go around New York city, not to mention it’s healthier and greener.

The way to get these bikes are pretty neat- You put in your credit card in the touch screen kiosk get your single or 3 days pass for $12 or $24 respectively plus taxes. These Docks and Kiosks can be found on every 10 blocks or so and a couple of avenues apart, the bike docking station map is available on their website and at the kiosk.

On browsing their [website](citibikenyc.com), I stumbled upon the system data — bunch of data from the parking docks mentioned earlier. They have made the data publicly available for explorers to analyse, visualise or just play with it. You can find the datasets for each quarter from May 2013 to Jun 2017 (at the time of writing of this post). In this post, we are using data from Jan-2015 to Jun-2017.

Enough talking let's get on with the code!

Load appropriate Data manipulation and Analysis library files
```python
# Load Data in a Pandas Dataframe
import pandas as pd
import seaborn as sns #just to make our visualization prettier ;-)
import numpy as np
import datetime
import math
%matplotlib inline
```
The dataset is split into monthly data files starting 2015 to 2017. Simple python code below will merge it into one DataFrame.
```python
frames = []
for i in range(0,19):
    temp_df = pd.read_csv('/Users/dgeek/Documents/fsdse/NY-Citibike/2015-2017-'+str(i+1)+'.csv')
    frames.append(temp_df)
# df is the master dataframe with all 19 csvs merged into one
# pd.concat creates a dataframe with the frames list object
df = pd.concat(frames)
```
Basic data cleaning tasks (self-explanatory if you are familiar with pandas or Python)
```python
# checks for first 5 values of dataframe - df
df.drop(['Unnamed: 0'],axis=1, inplace=True)
df.head()
# checks for length of DataFrame - df
print(len(df))
Creating a new dataframe ndf with selected (useful features)
ndf = df[['Trip_Duration_in_min', 'Start Time', 'Stop Time',
       'Start Station Name', 'End Station Name',
      'Bike ID', 'User Type','Birth Year', 'Gender']]
ndf.loc[:,('Birth Year')] = ndf['Birth Year'].astype(int)
ndf.head()
Treating Missing values

Since User Type is a categorical data field we will find the mode
UT_mode = ndf.mode().loc[:('User Type')]
# imputing mode inplace of missing values
ndf['User Type'].fillna(value=UT_mode['User Type'][0], inplace=True)
# check the counts again
ndf['User Type'].value_counts()
Out[92]:

Subscriber    688140
Customer       47362
Name: User Type, dtype: int64
Let's plot a pie chart to get a visual

In [203]:

fig = plt.figure(figsize=(3,3), dpi=200)
ax = plt.subplot(111)
df['User Type'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=270, fontsize=10)
```
Birth Year missing values
```python
BY_mode = ndf.mode().loc[:('Birth Year')]
ndf['Birth Year'].fillna(value=BY_mode['Birth Year'][0], inplace=True)
ndf.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 735502 entries, 0 to 735501
Data columns (total 9 columns):
Trip_Duration_in_min    735502 non-null int64
Start Time              735502 non-null object
Stop Time               735502 non-null object
Start Station Name      735502 non-null category
End Station Name        735502 non-null category
Bike ID                 735502 non-null category
User Type               735502 non-null category
Birth Year              735502 non-null int64
Gender                  735502 non-null category
dtypes: category(5), int64(2), object(2)
memory usage: 26.7+ MB
```
Converting Trip Duration in seconds to minutes
```python
df['Trip_Duration_in_min'] = df['Trip Duration']/60
# rounding to nearest decimal and converting to int32 for easier computation
df['Trip_Duration_in_min'] = df['Trip_Duration_in_min'].round().astype('int32')
```
Our dataset seems to be clean but we do not have enough number of features to start making predictions. Let’s do some Feature Engineering to fix this.

Convert start date and stop date to datatime dtype (datatype)
```python
ndf['Start Date'] = pd.to_datetime(ndf['Start Time'])
ndf['Stop Date'] = pd.to_datetime(ndf['Stop Time'])
Split start date feature variable column to derive meaningful features

ndf['Start Month'] = ndf['Start Date'].dt.month
ndf['Start Day'] = ndf['Start Date'].dt.day
ndf['Start Minute'] = ndf['Start Date'].dt.minute
ndf['Start Week'] = ndf['Start Date'].dt.minute
ndf['Start Weekday'] = ndf['Start Date'].dt.minute
ndf['Start Week'] = ndf['Start Date'].dt.week
ndf['Start Weekofyear'] = ndf['Start Date'].dt.weekofyear
ndf['Start weekday'] = ndf['Start Date'].dt.weekday
ndf['Start dayofyear'] = ndf['Start Date'].dt.dayofyear
ndf['Start weekday_name'] = ndf['Start Date'].dt.weekday_name
ndf['Start quarter'] = ndf['Start Date'].dt.quarter
do the same with Stop date feature column

ndf['Stop Month'] = ndf['Stop Date'].dt.month
ndf['Stop Day'] = ndf['Start Date'].dt.day
ndf['Stop Minute'] = ndf['Stop Date'].dt.minute
ndf['Stop Week'] = ndf['Stop Date'].dt.minute
ndf['Stop Weekday'] = ndf['Stop Date'].dt.minute
ndf['Stop Week'] = ndf['Stop Date'].dt.week
ndf['Stop Weekofyear'] = ndf['Stop Date'].dt.weekofyear
ndf['Stop weekday'] = ndf['Stop Date'].dt.weekday
ndf['Stop dayofyear'] = ndf['Stop Date'].dt.dayofyear
ndf['Stop weekday_name'] = ndf['Stop Date'].dt.weekday_name
ndf['Stop quarter'] = ndf['Stop Date'].dt.quarter
```
Let’s see what we have now
```python
ndf.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 735502 entries, 0 to 735501
Data columns (total 31 columns):
Trip_Duration_in_min    735502 non-null int64
Start Time              735502 non-null object
Stop Time               735502 non-null object
Start Station Name      735502 non-null category
End Station Name        735502 non-null category
Bike ID                 735502 non-null category
User Type               735502 non-null category
Birth Year              735502 non-null int64
Gender                  735502 non-null category
Start Date              735502 non-null datetime64[ns]
Stop Date               735502 non-null datetime64[ns]
Start Month             735502 non-null int64
Start Day               735502 non-null int64
Start Minute            735502 non-null int64
Start Week              735502 non-null int64
Start Weekday           735502 non-null int64
Start Weekofyear        735502 non-null int64
Start weekday           735502 non-null int64
Start dayofyear         735502 non-null int64
Start weekday_name      735502 non-null object
Start quarter           735502 non-null int64
Stop Month              735502 non-null int64
Stop Day                735502 non-null int64
Stop Minute             735502 non-null int64
Stop Week               735502 non-null int64
Stop Weekday            735502 non-null int64
Stop Weekofyear         735502 non-null int64
Stop weekday            735502 non-null int64
Stop dayofyear          735502 non-null int64
Stop weekday_name       735502 non-null object
Stop quarter            735502 non-null int64
dtypes: category(5), datetime64[ns](2), int64(20), object(4)
memory usage: 150.1+ MB
```
Looks like some of the features aren’t so useful (this has been explored in the Jupyter Notebook). Now let’s remove the feature which don’t seem relevant or useful.
```python
cols = ndf.columns
data = ndf[['Trip_Duration_in_min', 'Start Station Name',
       'End Station Name', 'User Type', 'Birth Year',
      'Start Month', 'Start Day', 'Start Minute',
       'Start Week', 'Start Weekofyear',
       'Start dayofyear', 'Start quarter', 'Stop Month',
       'Stop Day', 'Stop Minute',
       'Stop Weekofyear', 'Stop weekday', 'Stop dayofyear',
        'Stop quarter', 'Gender']]
data.head()
```
Selecting features is based on a lot of factors and it differs from one domain to another. In this use case, the features which did not add or give out useful information have been removed.
```python
data.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 735502 entries, 0 to 735501
Data columns (total 20 columns):
Trip_Duration_in_min    735502 non-null int64
Start Station Name      735502 non-null category
End Station Name        735502 non-null category
User Type               735502 non-null category
Birth Year              735502 non-null int64
Start Month             735502 non-null int64
Start Day               735502 non-null int64
Start Minute            735502 non-null int64
Start Week              735502 non-null int64
Start Weekofyear        735502 non-null int64
Start dayofyear         735502 non-null int64
Start quarter           735502 non-null int64
Stop Month              735502 non-null int64
Stop Day                735502 non-null int64
Stop Minute             735502 non-null int64
Stop Weekofyear         735502 non-null int64
Stop weekday            735502 non-null int64
Stop dayofyear          735502 non-null int64
Stop quarter            735502 non-null int64
Gender                  735502 non-null category
dtypes: category(4), int64(16)
memory usage: 92.6 MB
```
Notice that we have reduced features from 31 to 20. That is 11 x 735502 = 8,090,522 extra values, which the CPU(s) would have to churn while running the algorithm.

Now let’s get to the part we all were waiting for- Some ML magic!
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data = data.apply(le.fit_transform)
```
Label Encoding is a crucial step since algorithms only understand numeric values and our dataframe contains mostly categorical data, we use Sklearn’s Label Encoder from preprocessing module to convert them.
```
Splitting X (independent variables) and y values (dependent, values to be predicted)

```python
X = data.iloc[:-10000,:-1]
test_X = data.iloc[-10000:,:-1]
print(len(X), len(test_X))

725502 10000

y = data.iloc[:-10000,-1:]
test_y = data.iloc[-10000:,-1:]
print(len(y),len(test_y))

725502 10000
```
We will use  test_X & test_y later to see if our algorithm is actually performing.

```python
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

Out [168]: (544126, 19) (181376, 19) (544126, 1) (181376, 1)
```
Everything is in good shape.
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
kf = KFold(n_splits=3,random_state=2)
model = DecisionTreeClassifier()
results = cross_val_score(model, X, y, cv=kf)
print(results.mean()*100)
Out [169]: 99.6363896998
```
Pretty amazing accuracy score of 99.6363896998 % considering we did not tinker with any hyper parameters! But we have to ask if — there’s a data leakage or the model is just overfitting?

Let’s try predicting values on some unseen data.

```python
prediction = clf.predict(test_X)
pred_t = le.transform(prediction)
acc = (pred_t==test_y['Gender']).value_counts()
acc
Out[188]:

True     9964
False      36
Name: Gender, dtype: int64
In [199]:

print('Accuracy on Unseen data %f' %(acc[1]/10000*100)+' %')
Accuracy on Unseen data 99.640000 %
```
Wow! so the model did actually work! Does that mean I’m an ML Genius ? ! :D.

I wouldn’t flatter myself thinking about it B-). I’m just a rookie trying to learn the art and craft of Data Science, I may have made some mistakes along the way but slowly and steadily I’m learning something. If you find any errors, please feel free to correct me. I gladly accept constructive feedback.

Special Thanks to Mudassir Khan, Jash Shah, Mayuresh Shilotri, Shweta Doshi, Ankan Roy, Rohan Damodar, Gyanendra Dhanee & all my peers at GreyAtom School (These guys are amazing, they have helped me a lot in understanding Data Science & Machine Learning).

[Source Code on Github](https://github.com/nikhilakki/Predicting-the-Gender-of-the-riders-of-New-York-s-Citi-Bikes)

[Jupyter Notebook on Kaggle](https://www.kaggle.com/akkithetechie/new-york-city-bike-share-dataset)

References —

 1. [Decision Tree Classifier](http://mines.humanoriented.com/classes/2010/fall/csci568/portfolio_exports/lguo/decisionTree.html)
 2. [Decision Tree Classifier (Sci-kit learn)](http://scikit-learn.org/stable/modules/tree.html)
 3. [Hyper Parameters](https://www.quora.com/What-are-hyperparameters-in-machine-learning)
 4. [Overfitting](http://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/)
 5. [Leakage](https://www.kaggle.com/wiki/Leakage)

[Original Article on Medium](https://medium.com/@nikhilakki/predicting-gender-of-the-riders-of-new-yorks-citi-bikes-with-decision-trees-dcb169caad85)
