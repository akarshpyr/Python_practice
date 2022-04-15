#!/usr/bin/env python
# coding: utf-8

# In[1]:


a=234
print(id(a))


# In[4]:


print(a)
a


# In[5]:


2**4


# In[6]:


a=5+6j
type(a)


# In[7]:


a=8
type(a)


# In[8]:


a='shrink'
type(a)


# In[9]:


b=5.6778888888
type(b)


# In[10]:


a=5.89898989898989898989898989898989
type(a)


# In[12]:


d=3//2
print(d)


# In[13]:


d


# In[14]:


3//4


# In[15]:


a=True
type(a)


# In[16]:


print(type(a))


# In[22]:


a="+91"
b='9877623781'
print(a+ '' +b)


# In[23]:


print(a + " " + b)


# In[25]:


print(a+ ' ' +b[:5] + ' ' + b[5:])


# In[28]:


print(f"{a} {''} {b}")


# In[29]:


print(f"{a} {''} {b[:5]} {b[5:]}")


# In[30]:


a=4
id(a)


# In[31]:


a=8
id(a)


# In[32]:


a+a


# In[34]:


Ds+Python


# In[33]:


Ds= 10
print(Ds)
Python =30
print(Python)
Ds = "shunt"
print(Ds)
Ds = Python
print(Ds)


# In[35]:


Ds+Python


# In[36]:


oye = ["python", 234, "shut", 345, 3.5,8+7j,[3,4,4,5]]


# In[37]:


print(oye)


# In[38]:


oye[1:3]


# ###### oye[-1]

# In[40]:


oye[6][2]


# In[45]:


oyee = [1,2,3,4,[3,3,[3,3,4]]]
oyee[4][2][0]


# In[47]:


list1 = ["pyt",'pyth',3, 2018,[1,2,3,['d','e','f']]]
print(list1)


# In[51]:


list1[4][3][1]


# In[57]:


list3 = [2,3,4,5,6,290]
print(list3)
print(id(list3))
list3[3] = 233
list3
id(list3)
print(list3)
print(id(list3))


# In[59]:


list1 = [2,3,4,5,6]
a = [2,3.4,'oye']
b=[22,22,3,3,4,4,4,4]
print(list1,a,b)
print(list1 + a + b)
print(list1); print(a); print(b)


# In[14]:


list1 = [2013,1209,'sas',2013,1287,2233,3456,1234,2013,2678]
list_2013 = []
for i in range (0,len(list1)):
    if list1[i]==2013:
    list1.append(list_2013[i])


# In[17]:


alist = ['aka', 'aki',2100, 2344, ['zara',2900],'akarsh', 12990, 'meena']
print(alist)
alist.append(2013)
print(alist)
alist.pop()
print(alist)
alist.pop(2)
print(alist)
alist.pop(4)
print(alist)


# In[15]:


alist.insert(1,2022)


# In[10]:


print(alist)


# In[2]:


alist.pop(2)


# In[ ]:





# In[1]:


a = 13
print(a)


# In[22]:


list2 = [2021.222, 2.2, 234,2]
list3 = [12,13]
list2.extend(list3)
print(list2)
list3.extend(list2)
print(list3)
list2.reverse()
list2
print(list2)


# In[31]:


list3 = [123,232,435,22,33,333,33,'aki']
list3.remove('aki')
list3.sort()
print(list3)
print(list3)
list3.sort(reverse=True)
print(list3)
print(list3.count(33))
print(len(list3))


# In[32]:


a = [1,2,3,4,5]
b = 6 in a
c=2 in a
print(b)
print(c)
print(type(b))


# In[33]:


print(range(4))


# In[34]:


list(range(0,4))


# In[35]:


list(range(0,18,3))


# In[1]:


list1 = [2,2,3]
for m in list1:
    print(m**m)
    


# In[2]:


even_no =[]
list1 = [1,2,3,4,5,6,7,8,9,9,10]
for i in list1:
    if i%2==0:
        even_no.append(i)
print(even_no)


# In[12]:


a = list(range(1,501))
list2 = []
for i in a:
    if (i**0.5)%3 == 0:
        list2.append(int(i**0.5))
(print(list2))


# In[13]:


a1 = list(range(7,351,7))
b1=[]
for i in a1:
    if i%3==0:
        b1.append(i)
print(b1)
print(a1)


# In[2]:


tup1 = (12,23,45,'aki','oye',234)
tup2= (12,22,33,45)
print(tup1[2])
tup1 = (12,34,56)
print(tup1)


# In[3]:


tup3 = tup1 + tup2
print(tup3)


# In[4]:


tup2[2] = 34


# In[5]:


del(tup1)


# In[6]:


print(tup1)


# In[7]:


len(tup3)


# In[9]:


tup4 = tup3**4


# In[11]:


print(tup4)


# In[12]:


print(tup4)


# In[14]:


tup5 = tup4*2
print(tup5)


# In[21]:


a = {'aki': 'name', 'age':'26', 'quali': 'Mtech'}
print(a.get('quali','age'))


# In[22]:


print(a['age'])


# In[25]:


print(a.get('aki','age'))


# In[27]:


dic = {'ran1': list(range(9)), 'ran2': list(range(10))}
dic


# In[2]:


dic = {'name': 'aki', 'age': 26, 'dob':'1-10-1995'}
for i in dic:
    print(dic[i], end = ' ')


# In[1]:


get_ipython().system('pip install pandas')


# In[2]:


import pandas as pd


# In[5]:


dic['name'] = 'amrutha'
dic


# In[43]:


a = {'list1':[1,2,3],'list2':[4,5,6],'list3':[7,8,9]}
j = 0
for i in a.keys():
    print(a[i][j])
    j = j+1
del(a['list1'])
print(a)
a.clear()
print(a)


# In[33]:


for i in a.values():
    print(i[1])


# In[48]:


animals = {'aki', 'aki', 'dof','der'}
print('s:', animals)
print(animals)


# In[50]:


days = list(animals)
days


# In[51]:


ss = set(days)
ss


# In[2]:


a = 23
a


# In[3]:


get_ipython().system('pip install pandas')


# In[5]:


import pandas as pd


# In[6]:


df = pd.read_csv('https://raw.githubusercontent.com/Apress/data-analysis-and-visualization-using-python/master/Ch07/Salaries.csv')


# In[7]:


df


# In[8]:


mtcars = pd.read_csv('E:\\New folder\\mtcars.csv')


# In[9]:


mtcars


# In[10]:


mtcars.shape


# In[11]:


mtcars.head(5)


# In[12]:


mtcars.tail(4)


# In[13]:


df.head()


# In[14]:


df.tail()


# In[16]:


df.columns


# In[17]:


df.dtypes


# In[18]:


list(df.dtypes)


# In[19]:


df.describe()


# In[21]:


df[['phd']]


# In[23]:


df['phd'].mean()


# In[26]:


df[['salary']].describe()


# In[29]:


df.iloc[0]


# In[30]:


df.iloc[1:2,3:6]


# In[32]:


df.iloc[[0,5],[3,5]]


# In[39]:


df.iloc[3:4,0:4]


# In[42]:


df.median()


# In[43]:


df.quantile()


# In[45]:


df['phd'].max()


# In[46]:


df['salary'].quantile(0.75)


# In[52]:


round(df['salary'].mean(),3)


# In[55]:


df['salary'].value_counts()


# In[57]:


import warnings
warnings.filterwarnings('ignore')


# In[58]:


df.head(40).mean()


# In[2]:


import numpy as np


# In[3]:


a = np.arange(4)


# In[4]:


a


# In[5]:


s = np.arange(19,3,-2)


# In[6]:


s


# In[9]:


d = np.linspace(2,16,19)


# In[10]:


d


# In[13]:


f = (np.zeros(5))


# In[12]:


f


# In[14]:


c = np.zeros((3,4,2))


# In[15]:


c


# In[16]:


c[0,1]


# In[18]:


d= np.array([[1,2,3],[2,3,4],[2,3,4]])


# In[19]:


d


# In[20]:


d[:,1]


# In[21]:


d.max(axis=0)


# In[22]:


d.max(axis=1)


# In[23]:


d.min(axis=0)


# In[24]:


d.min(axis=1)


# In[25]:


d.sum(axis=0)


# In[27]:


d.sum(axis=1)


# In[28]:


np.sqrt(d)


# In[29]:


s = np.sqrt(d[0,1])


# In[30]:


s


# In[31]:


s = np.sqrt(d[:,1])


# In[32]:


s


# In[33]:


x = np.log(d)


# In[34]:


x


# In[35]:


x = np.square(d)


# In[36]:


x


# In[37]:


x = np.cumsum(d)


# In[39]:


x


# In[40]:


x = np.arange(51)


# In[41]:


x


# In[42]:


c = np.cumsum(x)


# In[43]:


c


# In[44]:


d= np.array([[1,2,3],[2,3,4],[2,3,4]])


# In[45]:


d


# In[46]:


e= np.array([[0,3,3],[44,3,4],[2,3,4]])


# In[47]:


e


# In[48]:


d+e


# In[49]:


d-e


# In[50]:


d*e


# In[51]:


d/e


# In[52]:


d//e


# In[54]:


s = np.hstack((d,e))


# In[55]:


s


# In[56]:


c = np.vstack((d,e))


# In[57]:


c


# In[58]:


c


# In[65]:


s = np.arange(1,26).reshape(5,5)


# In[66]:


s


# In[67]:


s


# In[68]:


d


# In[69]:


d


# In[70]:


df


# In[71]:


import pandas as pd


# In[72]:


df = pd.read_csv('https://raw.githubusercontent.com/Apress/data-analysis-and-visualization-using-python/master/Ch07/Salaries.csv')


# In[73]:


df


# In[74]:


df.iloc[1:3,3:4]


# In[76]:


df.iloc[[3,8],[5,3]]


# In[80]:


af = df.groupby(['salary']).mean()


# In[81]:


af


# In[83]:


df[['phd','salary']].agg(['min','max','mean'])


# In[88]:


df['phd'].mean()


# In[92]:


df1 = pd.read_csv('')


# In[93]:


mtcars = pd.read_csv('E:\\New folder\\mtcars.csv')


# In[95]:


df1 = mtcars


# df1 

# In[96]:


df1


# In[98]:


import matplotlib.pyplot as plt
import pandas as pd


# In[99]:


bc = pd.crosstab(mtcars.gear,mtcars.cyl)


# In[100]:


bc


# In[101]:


bc.plot(kind='bar')


# In[113]:


aes=mtcars[['gear']].value_counts()
aes


# In[111]:


aes.plot(kind='pie')


# In[108]:


sdf


# In[114]:


plt.scatter(mtcars.mpg, mtcars.qsec)


# In[117]:


plt.hist(mtcars['mpg'],facecolor = 'red', edgecolor = 'white', bins = 4)


# In[118]:


x = [2,3,8]
y= [3,4,5]


# In[119]:


plt.plot(x,y)
plt.xlim(0,10)
plt.ylim(0,10)
plt.xlabel('kgf2')
plt.ylabel('collections')
plt.title('coll data')
plt.suptitle('KFI',size = 20, y=1.04)


# In[122]:


plt.boxplot(mtcars['mpg'])


# In[123]:


plt.violinplot(mtcars['gear'])


# In[124]:


import seaborn as sns


# In[129]:


tips = sns.load_dataset('tips')


# In[130]:


tips


# In[134]:


sns.stripplot(y="tip", data=tips, jitter = True)
plt.ylabel('tips')
plt.show()


# In[141]:


sns.stripplot(y="tip", x="day", data = tips, jitter = True)
plt.ylabel('tips')
plt.xlabel('days')
plt.show()


# In[152]:


import warnings as warnings; warnings.filterwarnings('ignore')


# In[153]:


sns.swarmplot(y='tip',x='day', data=tips)
plt.xlabel('days');plt.ylabel('tip')
plt.show()


# In[154]:


sns.swarmplot(y='tip',x='day', data =tips,)


# In[155]:


sns.swarmplot(x='day',y='tip', data = tips, hue = 'sex')


# In[157]:


sns.swarmplot(x='day',y='tip',data = tips, hue='smoker')


# In[158]:


sns.boxplot(x='day', y='tip', data =tips)


# In[159]:


sns.violinplot(x='day', y='tip', data =tips)


# In[161]:


plt.subplot(1,2,1)
sns.boxplot(x='day', y='tip', data =tips)
plt.subplot(1,2,2)
sns.swarmplot(x='day', y='tip', data =tips)


# In[163]:


sns.violinplot(x='day', y='tip', data =tips, color = 'grey')
sns.stripplot(x='day', y='tip', data =tips)


# In[164]:


sns.jointplot(x='total_bill',y = 'tip', data = tips)


# In[165]:


sns.jointplot(x='total_bill',y = 'tip', data = tips, kind = 'kde')


# In[170]:


sns.pairplot(tips)


# In[171]:


sns.pairplot(tips,hue='sex')


# In[172]:


import pandas as pd


# In[181]:


data = pd.read_csv('E:\\New folder\\train.csv')


# In[182]:


data


# In[183]:


data.head()


# In[196]:


plt.subplot(1,2,1)
sns.boxplot(x='Survived', y='Age', data=data)
plt.subplot(1,2,2)
sns.boxplot(x='Sex',y='Age', data = data)


# In[199]:


data.groupby('Survived')['PassengerId'].count()


# In[201]:


sns.barplot(x='Sex',y='Survived',data=data)


# In[202]:


data.value_counts('Age')


# In[204]:


sns.boxplot(x='Sex',y='Age',hue='Survived',data=data)


# In[205]:


sns.barplot(x='Pclass',y='Survived', data=data, hue='Sex')


# In[213]:


ass = data.loc[data['Survived']==1,'Age'].dropna()


# In[214]:


sns.distplot(ass)


# In[215]:


ass1 = data.loc[data['Survived']==0,'Age'].dropna()


# In[216]:


sns.distplot(ass1)


# In[ ]:




