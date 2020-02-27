#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import ....
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
food =  pd.read_csv("food_nutrient_2011_13_AHS.csv", header=0,low_memory=False)


# In[2]:


#Question 1
#As shape fuction return the number of rows and columns also as the number of foods and attributes respectively
number_food=food.shape[0] 
print("Number of foods: {}".format(number_food))
number_attribute = food.shape[1]
print("Number of attributes: {}".format(number_attribute))
print(food.dtypes)


# In[3]:


#Question 2

#calculate the Magnesium's median
mag_median = food["Magnesium (Mg) (mg)"].median()
round_mag_median = round(mag_median,1)
print("Median value of Magnesium: {}".format(round_mag_median))

#calculate the Moisture's median
moi_mean = food["Moisture (g)"].mean()
round_moi_mean = round(moi_mean,1)
print("Mean value of Moisture: {}".format(round_moi_mean))


# In[4]:


#Question 3

# Part a
food["Survey ID"] = food["Survey ID"].astype(str)

# Part b
food_category = [i[0:2] for i in food["Survey ID"]]
food['Food Category'] = pd.Series(food_category)

# Part c
count= 0 
for digit in food['Food Category']:
    if digit == '13':
        count+=1
percent = count/5740*100
round_percent = round(percent,1)
print("% of foods which are Cereal based products and dishes (Food category 13) = {}".format(round_percent))


# In[5]:


#Question 4

#create 2 lists of Total fat (g) and Protein (g) that are "Cereal based products and dishes"
lst_fat = []
lst_protein = []
for num in range(number_food):
    if food['Food Category'][num] == '13':
        lst_fat.append(food['Total fat (g)'][num])
        lst_protein.append(food["Protein (g)"][num])
        
#create the DataFrame of Total fat (g) and Protein (g) that are "Cereal based products and dishes"
cereal_based = pd.DataFrame({'Total fat (g)': lst_fat,'Protein (g)' : lst_protein })

#plot the boxplots
boxplot = cereal_based.plot(kind = 'box')
plt.show()


# In[12]:


#Question 5
grouped = food.groupby("Food Category")
#create 2 list of "Food Category" and mean "Total sugars (g)"
category_lst = []
sugars_lst = []
for k, h in grouped:
    category_lst.append(k)
    sugars_lst.append(h["Total sugars (g)"].mean())

#create a dataframe contain the food category and the mean of total sugar 
cate_sugars_dict = {"Food Category" : category_lst,"Total sugars (g)": sugars_lst }
cate_sugars = pd.DataFrame(cate_sugars_dict)
#plot the barplots
cate_sugars.plot(kind='bar',x = "Food Category", y = "Total sugars (g)")
plt.show()


# In[7]:


#Question 6

#Part a

#create a list to store all the value of "Energy Power" 
energy_power_lst = []
for energy in food["Energy, with dietary fibre (kJ)"]:
    if energy > 1000:
        energy_power_lst.append(1)
    else:
        energy_power_lst.append(0)
food['Energy Power'] = pd.Series(energy_power_lst)
        

#Part b

#create 2 list to store the value of high energy and low energy
high_sugars=[]
high_protein=[]
low_sugars=[]
low_protein=[]
for num in range(number_food):
    if food["Energy Power"][num] == 1:
        high_sugars.append(food["Total sugars (g)"][num])
        high_protein.append(food["Protein (g)"][num])
    else:
        low_sugars.append(food["Total sugars (g)"][num])
        low_protein.append(food["Protein (g)"][num])
        
#plot the scatter plot       
plt.scatter(high_sugars,high_protein,color = 'red',label ='High')
plt.scatter(low_sugars,low_protein,color = 'blue',label='Low')
plt.xlabel("Total sugars (g)")
plt.ylabel("Protein (g)")
plt.legend()


# In[8]:


#Question 7

#Part a

#create 3 lists the store the normalised value of "Protein (g)", "Total fat (g)" and "Total sugars (g)"
new_protein_lst=[]
#find the maximum and minimum of protein
max_protein = max(food["Protein (g)"])
min_protein = min(food["Protein (g)"])
for old_protein in food["Protein (g)"]:
    new_protein = (old_protein-min_protein)/(max_protein-min_protein)
    new_protein_lst.append(new_protein)
    
new_fat_lst=[]
#find the maximum and minimum of total fat
max_fat = max(food["Total fat (g)"])
min_fat = min(food["Total fat (g)"])
for old_fat in food["Total fat (g)"]:
    new_fat = (old_fat-min_fat)/(max_fat-min_fat)
    new_fat_lst.append(new_fat)

new_sugars_lst=[]
#find the maximum and minimum of total sugars
max_sugars = max(food["Total sugars (g)"])
min_sugars = min(food["Total sugars (g)"])
for old_sugars in food["Total sugars (g)"]:
    new_sugars = (old_sugars-min_sugars)/(max_sugars-min_sugars)
    new_sugars_lst.append(new_sugars)
#Part b
#create a list of low and high energy
energy_lst=[]
for i in range(number_food):
    if food["Energy, with dietary fibre (kJ)"][i] <= 1000:
        energy_lst.append("Low Energy")
    else:
        energy_lst.append("High Energy")
normalised_dict = {"Energy":energy_lst,"Total sugars (g)":new_sugars_lst,"Total fat (g)":new_fat_lst,"Protein (g)":new_protein_lst}
normalised = pd.DataFrame(normalised_dict)      
pd.plotting.parallel_coordinates(normalised,'Energy',color=('blue','red'))
plt.show()


# In[15]:


#Question 8


#Part a
calorie_count =[] 
for num in range(number_food):
    calorie = 4*food["Protein (g)"][num]+4*food["Available carbohydrates, with sugar alcohols (g)"][num]+9*food["Total fat (g)"][num]+7*food["Alcohol (g)"][num]
    calorie_count.append(calorie)
food['calorie count per 100g'] = pd.Series(calorie_count)

#Part b
sorted_calorie = food.sort_values(by = 'calorie count per 100g',ascending = False)
time =0
for i in sorted_calorie.index[:5]:
    time += 1
    print("{}.{food} {calorie}".format(time,food=food["Food Name"][i],calorie=round(food['calorie count per 100g'][i],1)))
    
#Part c
grouped = food.groupby("Food Category")
#create 2 lists contain the value of category and mean of calorie
category_lst = []
calorie_lst = []
for k, h in grouped:
    category_lst.append(k)
    calorie_lst.append(h['calorie count per 100g'].mean())
#create a dictionary of food category and mean of calorie
cate_calorie_dict = {"Food Category" : category_lst,"Mean of total calorie (g)": calorie_lst }
cate_calorie = pd.DataFrame(cate_calorie_dict)
#plot the pie chart

plt.pie(cate_calorie["Mean of total calorie (g)"], labels=cate_calorie["Food Category"],autopct='%1.1f%%',startangle=90)
plt.axis('equal') 
plt.show() 


# In[10]:


#Question 9

#Part a
#open the file contains the values of classification
classification =  pd.read_csv("8i. Australian Health Survey Classification System.csv", header=0,low_memory=False)
number_class = classification.shape[0]
#create a list contains the class of food category
category_name_lst=[]
for i in range(number_food):
    for j in range(number_class):
        if str(classification["Food Group Code"][j]) == food["Food Category"][i]:
            category_name_lst.append(classification["Food Group and Sub-Group Name"][j])
food['Food Category Name'] = pd.Series(category_name_lst)

#Part b
#create a dictionary of food category and mean of total sugars. 
import json
grouped = food.groupby("Food Category Name")
dic={}
for k, h in grouped:
    dic[k] = round(h['Total sugars (g)'].mean(),1)
#make the output in json format
print (json.dumps({"Mean total sugars (g) by category": dic}, indent=2))


# In[ ]:




