#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# In[2]:


from scipy.spatial.distance import pdist, squareform
def VAT(R):
    R = np.array(R)
    N, M = R.shape
    if N != M:
        R = squareform(pdist(R))
    J = list(range(0, N))
    
    y = np.max(R, axis=0)
    i = np.argmax(R, axis=0)
    j = np.argmax(y)
    y = np.max(y)
    
    I = i[j]
    del J[I]
    
    y = np.min(R[I,J], axis=0)
    j = np.argmin(R[I,J], axis=0)
    I = [I, J[j]]
    J = [e for e in J if e != J[j]]
    C = [1,1]

    for r in range(2, N-1):
        y = np.min(R[I,:][:,J], axis=0)
        i = np.argmin(R[I,:][:,J], axis=0)
        j = np.argmin(y)
        y = np.min(y)
        I.extend([J[j]])
        J = [e for e in J if e != J[j]]
        C.extend([i[j]])

    y = np.min(R[I,:][:,J], axis=0)
    i = np.argmin(R[I,:][:,J], axis=0)

    I.extend(J)
    C.extend(i)

    RI = list(range(N))
    for idx, val in enumerate(I):
        RI[val] = idx

    RV = R[I,:][:,I]
    return RV.tolist(), C, I


def entropy(probs):
    
    return -probs.dot(np.log2(probs))


def mutual_info(df):
    
    Hx = entropy(df.iloc[:,0].value_counts(normalize=True, sort=False))
    Hy = entropy(df.iloc[:,1].value_counts(normalize=True, sort=False))
    
    counts = df.groupby(list(df.columns.values)).size()
    probs = counts/ counts.values.sum()
    H_xy = entropy(probs)

    # Mutual Information
    I_xy = Hx + Hy - H_xy
    MI = I_xy
    NMI = I_xy/min(Hx,Hy) #I_xy/np.sqrt(H_x*H_y)
    
    return {'H_'+list(df)[0]:Hx,'H_'+list(df)[1]:Hy,'MI':MI,'NMI':NMI} 


# In[3]:


food =  pd.read_csv("food_nutrient_2011_13_AHS.csv", header=0,low_memory=False)


# In[4]:


##Question 1


###1a
continuous_feature = pd.DataFrame(food.iloc[:,4:57])

###1b
scaler = StandardScaler()
foodscaled = scaler.fit_transform(continuous_feature)



###1c
print("Q1.c : foodscaled matrix details")
print("Number of rows:",foodscaled.shape[0])
print("Number of columns:",foodscaled.shape[1])
print("Min:",round(foodscaled.min(),1))
print("Max:",round(foodscaled.max(),1))
print("Mean:",round(foodscaled.mean(),1))
print("Standard Deviation:",round(foodscaled.std(),1))



# In[5]:


#Question 2


###2a
food['EnergyLevel'] = np.where(food['Energy, with dietary fibre (kJ)']>1000, 1, 0)


###2b
pca = PCA(n_components=2)
foodreduced = pca.fit_transform(foodscaled)
###2c
reduced = pd.DataFrame({"1st principle component":[i[0] for i in foodreduced],"2nd principle component":[i[1] for i in foodreduced]
                       ,'EnergyLevel':food['EnergyLevel']})
food_high=reduced.loc[reduced['EnergyLevel']==1]
food_low=reduced.loc[reduced['EnergyLevel']==0]

fig, ax = plt.subplots()

ax.scatter(food_high["1st principle component"],food_high["2nd principle component"],color='red',label='High energy',s=1.5)
ax.scatter(food_low["1st principle component"],food_low["2nd principle component"],color='blue',label='Low energy',s=1.5)
leg = ax.legend()
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.title('Scatter plot about 1st PCs vs 2nd PCs based on Energy Level')
plt.show()


# Question 2d answer here
# 
# According to the scatter plot, we can see most of the points represented for 'high energy' position together in groups and so do most of the points represented for 'low energy'. 
# 
# The 2 groups also set apart which can make us easily to identify the food according to their 'Energy Level' characteristic. Based on the plot, we can also visualize which kinds of food set far away from their group which may suggest there are something special about them. Also we can see while both of them have the same value for 2nd PCs but the 'High Energy' one have higher value in 1st PCs.
# 
# The advantages of PCA for this food dataset. As this food dataset has total 53 attributes and 5740 values for each attribute and we can not interpret such a high dimensional data like this so PCA will reduce the dimensionality of data, while keeping as much variation as possible. Thanks to the first principle component and second principle component PCA produced we have a summary of food with the 'Energy Level' characteristic so that we can easily visualize their behaviors.
# 
# The disadvantages: 
# PCA assumes that the principle components are orthogonal and if the dataframe is not linear it may not work. ALso, sometime, the results are not always good enough for visualization and make it difficult to interpret, for example, in some cases, both groups maybe mix together so we can not get any usefull information.

# In[6]:


#Question 3


###3a
food['Survey ID']=food['Survey ID'].astype(str)
food['Food Category'] = food['Survey ID'].str.slice(0,2)

###3b
food_row = food.shape[0]
foodscaledsample=[]
for i in range(food_row):
    if food['Food Category'][i] == '13' or food['Food Category'][i] == '20' or food['Food Category'][i] == '24':
        foodscaledsample.append(foodscaled[i])
        
###3c
RV, C, I = VAT(foodscaledsample)


###3d
x = sns.heatmap(RV, cmap='gist_stern', xticklabels=False, yticklabels=False)
x.set(xlabel='Objects', ylabel='Objects')
plt.title('Heatmap of foodscaledsample')
plt.show()


# Question 3e here
# There a 3 main clusters in the heatmap.This is expected because we select only foods from 3 categories: 13, 20 and 24.The use of different colormaps produce visualisations of varying usefulness because the most of value represented in the heatmap is between 0 and 8 and this color map have 2 distinguished color in this boundary while most of other colormaps have only one color in this boundary so we can not visualize the different if we use these colors. So to visualize the cluster in this task, we have to choose the colormap has 2 distinguished colors in the low range which between 0 and 8, for example the 'jet' colormap is also a good alternative.

# In[7]:


#Question 4

###4a 

#use the VAT of foodscaledsample in this question because it take too long to produce the value of VAT
#for the whole foodscaled sample
sum_lst=[]
for i in range(2,26):
    kmeans = KMeans(n_clusters=i,random_state=100).fit(RV)
    sum_lst.append(kmeans.inertia_)

plt.plot(range(2,26),sum_lst)
plt.title('Graph compare SSE vs k-values')
plt.show()


# Question 4b answer here The plot suddently drop at k = 3 and then stay almost stable for the rest value of k. The elbow at k=3 this is expected because when the number of cluster increase,the distance between each points and the centroid will decrease so the errors will decrease.

# In[8]:


###Question 5


###5a
first10_nuitri = food.iloc[:,4:14]
correlation_matrix=first10_nuitri.corr(method='pearson')

sns.heatmap(correlation_matrix, cmap='viridis')
plt.title('Pearson correlation heatmap of first 10 nuitrion')
plt.show()


# In[9]:


###5b code

# weite funciton to generate the binning part for each pairs
def find_bins(feature,num_bins):
    lst_feature = list(feature)
    sorted_feature=sorted(lst_feature)
    width_feature=(max(sorted_feature)-min(sorted_feature))/num_bins
    boundaries_feature=[]
    bin_feature=[]
    for i in range(num_bins+1):
        boundaries_feature.append(round(min(sorted_feature)+width_feature*i,3))
    for value in sorted_feature:
        for j in range(num_bins):
            if boundaries_feature[j]<= value < boundaries_feature[j+1]:
                bin_feature.append(j+1)    
        if value == boundaries_feature[num_bins]:
            bin_feature.append(num_bins)
    return bin_feature

MI_lst=[]
bins=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
for num_bins in bins:
    bin_protein= find_bins(food['Protein (g)'],num_bins)
    bin_energy = sorted(find_bins(food['Energy, with dietary fibre (kJ)'],num_bins),reverse=True)
    df = pd.DataFrame({'Protein (g)':bin_protein,'Energy, with dietary fibre (kJ)':bin_energy})
    MI_lst.append(mutual_info(df)['MI'])
    
plt.plot(bins,MI_lst)
plt.xlabel('Numbers of bins')
plt.ylabel('MI of the feature pair')
plt.title('Graph compare MI vs numbers of bins')
plt.show()


# Question 5b comment here
# 
# While the number of bins increase, the MI value of feature pair also increase at the high rate at the start and then lower rate in the remaining value

# In[11]:


###5c code
#top 10 correlation pairs
all_nuitri= food.iloc[:,4:57]
full_correlation_matrix=all_nuitri.corr(method='pearson')
unstack_correlation = full_correlation_matrix.unstack().abs()
top10_correlation_pairs = unstack_correlation.sort_values(ascending=False,kind="quicksort").drop_duplicates()[1:11]



# In[12]:


#top 10 MI with 20 equal_width bins
num_bins=20
lst=[]
for attribute_1 in all_nuitri:
    inner_lst=[]
    for attribute_2 in all_nuitri:
        if attribute_1 == attribute_2:
            inner_lst.append(None)
        else:
            bin1=find_bins(food[attribute_1],num_bins)
            bin2=sorted(find_bins(food[attribute_2],num_bins),reverse=True)
            df = pd.DataFrame({attribute_1:bin1,attribute_2:bin2})
            inner_lst.append(mutual_info(df)['MI'])
    lst.append(inner_lst)

df_bin = pd.DataFrame(data=lst,index=list(food)[4:57],columns= list(food)[4:57])
unstack_bin = df_bin.unstack()
top_MI_pairs = unstack_bin.sort_values(ascending=False,kind="quicksort")


# In[13]:


# use [:20:2] to drop the duplicated numbers for MI
print("Top 10 MI pairs:")
print(top_MI_pairs[:20:2])
print("")
#
print("Top 10 correlation pair:")
print(top10_correlation_pairs)


# Question 5c comment here
# 
# They are all different because the MI can detect both linear and non linear dependencies (unlike
# Pearson) and MI is also applicable and very effective for use with discrete features (unlike Pearson correlation)

# In[14]:


##Question 6


###6a

X_train, X_test, y_train, y_test = train_test_split( foodscaled, food['Food Category'], train_size=0.8, test_size=0.2)
print("Q6.a: Train Test Split Results")
print('X_train matrix:',X_train.shape[0],X_train.shape[1])
print('y_train labels:',y_train.shape[0])
print('X_test matrix:',X_test.shape[0],X_test.shape[1])
print('y_test labels:',y_test.shape[0])
###6b

splits = range(1,41)
accu_list=[]
for split in splits:
    dt = DecisionTreeClassifier(criterion="entropy",random_state=1,max_depth=split)
    dt.fit(X_train, y_train)
    y_pred=dt.predict(X_test)
    accu_list.append(accuracy_score(y_test, y_pred))

plt.plot(splits,accu_list)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Graph between Accuracy vs Max Depth')
plt.show()


# Question 6c comment here
# The shape of this graph is increase at very high rate at the start, when it reach the peaks, there is a slight drop and then it is stable for the rest. The local peak at 12 approximately. 
# 
# This graph have this shape because the maximum depth is where we allow the tree to grow to.So this plot have this shape because when we set the max depth too low, the decision tree have too little flexibility to capture the patterns and interactions in the training data and this will decrease the accuracy.So if we set max_depth higher, the decision tree might fit the training data and produce the higher accuracy.

# In[15]:


#Question 7

###7a
print('k=1')
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred=knn.predict(X_train)
print('Train accuracy: ',round(accuracy_score(y_train, y_pred),1))

y_pred=knn.predict(X_test)
print('Test accuracy: ',round(accuracy_score(y_test, y_pred),1))

##7b
print('k=3')
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred=knn.predict(X_train)
print('Train accuracy: ',round(accuracy_score(y_train, y_pred),1))

y_pred=knn.predict(X_test)
print('Test accuracy: ',round(accuracy_score(y_test, y_pred),1))


# Question 7c comment here
# The test accuracy are the same as 0.9 but the train accuracy when k=1 is 1.0 which is higher than 0.9 which is the train accuracy when k=3. This is because when K is small, we are make the region smaller . So a small value for K provides the most flexible fit, which have low bias but high variance. Larger values of K will have bigger decision boundaries which means lower variance but increased bias.
# 
# Question 7d comment here
# The 90% estimate would be biased, since the testing data (Food Category info) was looked at when doing feature selection. This provided information to the feature selection process that should not have been seen and make the data leaked. Consequently the model that was trained using the results from the feature selection was developed on information that should not have been seen. The reported accuracy will thus likely be over optimistic. So we can reduce the data training set solved this problem because it will limit the  data leaks.

# In[16]:


#Question 8 code


# use the clustering labels method

#create a datafram contain the foodscaled value
df_foodscaled = pd.DataFrame(data=foodscaled,index=range(5740),columns= list(food)[4:57])
kmeans = KMeans(n_clusters=5).fit(foodscaled)

#add the value of cluser label to dataframe according to its food
df_foodscaled['Cluster Label']=kmeans.labels_

# use the interaction term pairs method

#add the value of interaction term pair using the f_pair = f1*f2
for i in range(53):
    for j in range(i+1,53):
        value = df_foodscaled.iloc[:,i]*df_foodscaled.iloc[:,j]
        df_foodscaled['F({}) vs F({})'.format(i,j)] = value


df_foodscaled = df_foodscaled.round(3)

#create the MI value between each feature from df_foodscaled and the class label

#let use 20 equal-width bins and find top MI values between each attribute and class label
num_bins=20
lst=[]
food['Food Category'] = food['Food Category'].astype('float64')
for attribute_1 in df_foodscaled:
        bin1=find_bins(df_foodscaled[attribute_1],num_bins)
        bin2=sorted(find_bins(food['Food Category'],num_bins),reverse=True)
        df = pd.DataFrame({attribute_1:bin1,'Food Category':bin2})
        lst.append(mutual_info(df)['MI'])
df_bin_foodscaled = pd.DataFrame(data=lst,index=list(df_foodscaled),columns= ['Food Category'])
unstack_bin_foodscaled = df_bin_foodscaled.unstack()
top_MI_pairs_foodscaled = unstack_bin_foodscaled.sort_values(ascending=False,kind="quicksort")


#choose n = 50 to select the top50 
top50_MI_pairs_foodscaled = top_MI_pairs_foodscaled[:50]

top50_2d= [[i] for i in top50_MI_pairs_foodscaled]
#find the accuracy using k-NN with this information

#using the same training and test instances to compare with Q7
X_train, X_test, y_train, y_test = train_test_split( top50_2d , food['Food Category'][:50], train_size=0.8, test_size=0.2)

#try k=1
print('k=1')
knn = KNeighborsClassifier(n_neighbors=1)


knn.fit(X_train, y_train)

y_pred=knn.predict(X_train)
print('Train accuracy: ',round(accuracy_score(y_train, y_pred),1))

y_pred=knn.predict(X_test)
print('Test accuracy: ',round(accuracy_score(y_test, y_pred),1))

#try k=3
print('k=3')
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred=knn.predict(X_train)
print('Train accuracy: ',round(accuracy_score(y_train, y_pred),1))

y_pred=knn.predict(X_test)
print('Test accuracy: ',round(accuracy_score(y_test, y_pred),1)) 


# 8.ii comment:
# The feature generation and selection methods can deliver a boost in prediction accuracy compared to using k-NN just on the original features in foodscaled. For this example, we can see the accuracy increase from 0.9 to 1. However, it is also not very effective because it is time- consuming when generate the additional features and compute the MI for each pair.
# In general, I think this method will work well with small dataset.

# In[ ]:





# In[ ]:




