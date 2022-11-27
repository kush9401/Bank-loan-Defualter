#!/usr/bin/env python
# coding: utf-8

# # EDA Case Study- Assignment
# 

# ### Introduction
# This Case study will help us to establish an understaning of real business problems in the banking sector.

# 
# 
# ### Problem Statement: 
# There are two types of risks associated with Bank Desisions on giving loans
# 
# 1). If the applicant is likely to repay the loan and not approving the loan will result in loss of Business.
# 
# 2). If the applicant doesn't repay the loans, he/she is likely to be a defaulter- which is a financial loss for the company.
# 
# Identifying these two is an important aspcect in making business decisions and the company wants to understand the driving force behind loan default.
# 
# 

# ### Objective :
# 
# 1. To reduce the defaulters.
# 2. To highlight the potential customers
# 3. To understand what factors controls the loan defaulters
# 4. To make business decision as a company which can reduce financial loss.
# 
# The data provided below have a lot of information about the application at the time of loan processing. 

# ###  Importing the Libraries

# In[744]:


#importing the warnings.To ignore unwanted warnings in plots
import jovian
import warnings
warnings.filterwarnings('ignore')


# In[745]:


#importing libraries.
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
pd.set_option('display.max_columns',150) #For showing large data


# ### Loading the Data

# In[746]:


#read the data set of "Previous_application".
prev= pd.read_csv('previous_application.csv')
#read the data set of "columns_description"
df=pd.read_csv('columns_description.csv' ,sep=";", encoding='cp1252') 
#read the data set of "application_data"
appl= pd.read_csv('application_data.csv')


# #### Checking the structure of the data.

# In[747]:


#Checking the head
appl.head()


# In[748]:


#Checking the tail
appl.tail()


# In[749]:


#Checking Summary
appl.info(verbose=True)


# In[750]:


appl.describe()


# In[751]:


#Checking Shape
appl.shape


# ### Data Cleaning

# a. Quality Check

# In[752]:


# It is a very large data as it has 122 columns and over 3 lakhs rows. 


# In[753]:


#Checking the missing values in the Application Data
appl.isnull().sum()


# #### b. Checking Missing values

# In[754]:


# Checking % of Missing values
appl.isnull().sum()*100/len(appl)


# In[755]:


#We have setup a random cutoff of 45%, which means a column with more than 45% of missing values will be removed from our data as they can severy affect our data analysis.
appl_miss_col=appl.isnull().sum()*100/len(appl)
appl_miss_col=appl_miss_col[appl_miss_col.values>45]
print(appl_miss_col)


# In[756]:


#Checking number of columns which has missing values more than 45%.
len(appl_miss_col)


# In[757]:


# Dropping these 49 columns
appl_miss_col=list(appl_miss_col[appl_miss_col.values>=45].index)
appl.drop(labels=appl_miss_col,axis=1,inplace=True)


# In[758]:


# Checking the shape of the data to confirm that the columns have been dropped
appl.shape


# In[759]:


#Now we will check the columns which has less than 13.5% of missing values(This number came from analysing the missing values in the data)
appl_miss_col=appl.isnull().sum()*100/len(appl)
appl_13_mv=appl_miss_col[appl_miss_col.values<=13.5]


# In[760]:


appl_13_mv.shape


# In[761]:


# Checking the summary again
appl.info()


# In[762]:


#From analysing the daya, we can say that'FLAG_DOCUMENT_2 to 21' doesn't have a significance and will be of no use in the data interpretation. 
#Hence, we will further drop these columns
appl_drop_flag=list(['FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21'])


# In[763]:


appl_drop_flag


# In[764]:


appl.drop(appl_drop_flag,axis=1,inplace=True)


# In[765]:


#Checking the summary again
appl.info()


# In[766]:


# Now we will check important columns which has missing values <13.5% and their values
appl.AMT_REQ_CREDIT_BUREAU_DAY.value_counts(normalize=True)

This table shows that in 'AMT_REQ_CREDIT_BUREAU_DAY' column, 99.44% values are 0. Since, it is a vast number, so we can fill up 0 as the missing values in these columns
# In[767]:


# Filling up the missing values with 0
appl.AMT_REQ_CREDIT_BUREAU_DAY.fillna(0,inplace=True)


# In[768]:


# Checking the 'AMT_REQ_CREDIT_BUREAU_HOUR' column
appl.AMT_REQ_CREDIT_BUREAU_HOUR.value_counts(normalize=True)


# In[769]:


# Filling up the missing values with 0 again as 99.3% values are 0
appl.AMT_REQ_CREDIT_BUREAU_HOUR.fillna(0,inplace=True)


# In[770]:


# Checking the 'AMT_REQ_CREDIT_BUREAU_MON' column
appl.AMT_REQ_CREDIT_BUREAU_MON.value_counts(normalize=True)


# In[771]:


# Filling up the missing values with 0 again as 83.5% values are 0
appl.AMT_REQ_CREDIT_BUREAU_MON.fillna(0,inplace=True)


# In[772]:


# Checking the 'AMT_REQ_CREDIT_BUREAU_QRT' column
appl.AMT_REQ_CREDIT_BUREAU_QRT.value_counts(normalize=True)


# In[773]:


# Filling up the missing values with 0 again as 80.9% values are 0
appl.AMT_REQ_CREDIT_BUREAU_QRT.fillna(0,inplace=True)


# In[774]:


# Checking the 'AMT_REQ_CREDIT_BUREAU_WEEK' column
appl.AMT_REQ_CREDIT_BUREAU_WEEK.value_counts(normalize=True)


# In[775]:


# Filling up the missing values with 0 again as 96.7% values are 0
appl.AMT_REQ_CREDIT_BUREAU_WEEK.fillna(0,inplace=True)


# In[776]:


# Checking the 'AMT_REQ_CREDIT_BUREAU_YEAr' column
appl.AMT_REQ_CREDIT_BUREAU_YEAR.value_counts(normalize=True)

This AMT_REQ_CREDIT_BUREAU_YEAR column, we cannot replace as only 26% of the values are 0
# In[777]:


# Checking the columns having less null percentage
appl.isnull().sum()/len(appl)*100


# ###### 1. Analysis of AMT_REQ_CREDIT_BUREAU_YEAR column
# 
# 

# In[778]:


#plotting the values of AMT_REQ_CREDIT_BUREAU_YEA column to detect outliers
plt.figure(figsize=(6,7))
sns.boxplot(y=appl['AMT_REQ_CREDIT_BUREAU_YEAR'])
plt.title("AMT_REQ_CREDIT_BUREAU_YEAR",fontsize=13)
plt.show()

This boxplot shows that there are many outliers. hence, we need to check the mean or median
# In[779]:


#Checking mean, median
print(appl['AMT_REQ_CREDIT_BUREAU_YEAR'].describe())


# In[780]:


#Checking the missing value in these columns.
appl['AMT_REQ_CREDIT_BUREAU_YEAR'].isnull().sum()


# In[781]:


#Imputing the missing values with median
appl_arcby_m=appl['AMT_REQ_CREDIT_BUREAU_YEAR'].median()
appl['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(value=appl_arcby_m,inplace=True)


# In[782]:


#Checking the missing value in these columns.
appl['AMT_REQ_CREDIT_BUREAU_YEAR'].isnull().sum()


# ##### 2. Analysis of CNT_FAM_MEMBERS

# In[783]:


#checking count of family members
appl['CNT_FAM_MEMBERS'].value_counts(dropna=False)


# In[784]:


#plotting the values of CNT_FAM_MEMBERS column using box plot to detect outliers
sns.boxplot(y=appl['CNT_FAM_MEMBERS'])
plt.title("Analysis of CNT_FAM_MEMBERS",fontsize=13)
plt.show()

There are no outliers present in this column 
# In[785]:


#Describing the info of this column

print(appl['CNT_FAM_MEMBERS'].describe())


# In[786]:


#Imputing the missing values with median
appl_cfm=appl['CNT_FAM_MEMBERS'].median()
appl.CNT_FAM_MEMBERS.fillna(appl_cfm,inplace=True)


# In[787]:


#checking the count of missing value for CNT_FAM_MEMBERS column
appl.CNT_FAM_MEMBERS.isnull().sum()


# ##### 3. Analysing AMT_GOODS_PRICE Column

# In[788]:


#checking count of family members
appl['AMT_GOODS_PRICE'].value_counts(dropna=False)


# In[789]:


# Checking the missing values
appl['AMT_GOODS_PRICE'].isnull().sum()


# In[790]:


#plotting the values of AMT_GOODS_PRICE column
plt.figure(figsize=(8,6))
sns.boxplot(y=appl['AMT_GOODS_PRICE'])
plt.title("Analysis of AMT_GOODS_PRICE",fontsize=13)
plt.show()


# In[791]:


#describe the info of column AMT_GOODS_PRICE

print(appl['AMT_GOODS_PRICE'].describe())

Inference: All the values doesn't conclude that we can impute the missing values in this column and logically we have to keep the missing values.

# ##### 4. Analysis of CODE_GENDER Column
# 

# In[792]:


#checking count of Gender for Male and Female
appl['CODE_GENDER'].value_counts(dropna=False)

Clearly we can see that there are only 4 rows with XNA and Females are the majority. So, there would be no imapct when we update the XNA to F
# In[793]:


# Accesing the rows with XNA using Loc function
appl.loc[appl['CODE_GENDER']=='XNA','CODE_GENDER']='F'


# In[794]:


#Checking the vale counts again in percentage
appl['CODE_GENDER'].value_counts(dropna=False)*100/len(appl)


# In[943]:


(appl['CODE_GENDER'].value_counts(dropna=False)*100/len(appl)).plot.bar(title='Analysis of Gender Code')
plt.xticks(rotation=0)
plt.show()


# ##### 5. Analysis of AMT_ANNUITY column

# In[797]:


#checking count of Annuity
appl['AMT_ANNUITY'].value_counts(dropna=False)


# In[798]:


#plotting the values of AMT_ANNUITY' column
plt.figure(figsize=(8,6))
sns.boxplot(y=appl['AMT_ANNUITY'])
plt.title("Analysis of AMT_ANNUITY'",fontsize=13)
plt.show()


# In[799]:



print(appl['AMT_ANNUITY'].describe())

Inference: We can clearly see that maximum value is quite higher. Hence, we take median to replace the value to predict some insights from the data
# In[800]:


#Imputing the median value
appl_ann=appl['AMT_ANNUITY'].median()
appl['AMT_ANNUITY'].fillna(value = appl_ann, inplace =True)


# In[801]:


appl['AMT_ANNUITY'].isnull().sum()


# ##### 6. Analyze DAYS_BIRTH column
# 

# In[802]:


#checking count of DAYS_BIRTH
appl['DAYS_BIRTH'].value_counts(dropna=False)


# In[803]:


#plotting the values of 'DAYS_BIRTH' column
plt.figure(figsize=(8,6))
sns.boxplot(y=appl['DAYS_BIRTH'])
plt.title("Analysis of DAYS_BIRTH'",fontsize=13)
plt.show()


# Inference: There is nothing unusual here but days birth aren't seemed to be in correct format

# In[804]:


appl.DAYS_BIRTH.describe()


# In[805]:


# Adding a new column AGE
appl['AGE'] = np.ceil(appl.DAYS_BIRTH / -365)
appl.AGE.describe()

Everyting looks fine in this. Maximum age is 70. Hence, no outliers
# ............................................

# In[806]:


#Converting all continous colums to numeric so that inference is easy 
num_col=['TARGET','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','REGION_POPULATION_RELATIVE','DAYS_BIRTH',
                'DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','HOUR_APPR_PROCESS_START','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']

appl[num_col]=appl[num_col].apply(pd.to_numeric)
appl.head(5)

#### Since, we have more Columns of negative value.. So we will correct them
# ##### 7. Correction for DAYS_EMPLOYED column

# In[807]:


# Checking Mean, Median 
print(appl.DAYS_EMPLOYED.describe())


# In[808]:


# Adding a new column TIME_EMPLOYED to round off and removing the negative value
appl['TIME_YEARS_EMPLOYED'] = np.round(appl.DAYS_EMPLOYED / -365,1)
appl.TIME_YEARS_EMPLOYED.describe()


# In[809]:


#Still there are some negative values. So, we will replace it
appl[appl['TIME_YEARS_EMPLOYED']<0].TIME_YEARS_EMPLOYED.value_counts()

We can clearly see that for 55374 rows, there is a same value which is -1000.7, which seems to be irrelevant. Hence, we should convert it to NaN
# In[810]:


appl.loc[appl['TIME_YEARS_EMPLOYED'] < 0, 'TIME_YEARS_EMPLOYED'] = np.NaN
appl.loc[appl['TIME_YEARS_EMPLOYED'] < 0, 'TIME_YEARS_EMPLOYED'].value_counts()


# ##### 8. Correction for DAYS_REGISTRATION & DAYS_ID_PUBLISH column

# In[811]:


#Correcting the values from Positive to Negative
appl['DAYS_REGISTRATION']=abs(appl['DAYS_REGISTRATION'])
appl['DAYS_ID_PUBLISH']=abs(appl['DAYS_ID_PUBLISH'])


# In[812]:


appl.head()


# ##### 9. Analyses of AMT_INCOME_TOTAL Column

# In[813]:


appl.AMT_INCOME_TOTAL.value_counts()


# In[814]:


#plotting the values of 'AMT_INCOME_TOTAL' column
plt.figure(figsize=[10,5])
sns.boxplot(y=appl.AMT_INCOME_TOTAL)
plt.title("Analysis of AMT_INCOME_TOTAL",fontsize=13)
plt.show()

Checkiing for outliers present in the data and 1 value is pretty high in the data
# In[815]:


appl.AMT_INCOME_TOTAL.describe()


# In[816]:


appl.AMT_INCOME_TOTAL.quantile([.5, .7, .9, .95, 0.99, 0.999, 0.9999])


# In[817]:


#Checking number of columns greater than 99th percentile for AMT_INCOME_TOTAL column
appl_ti_drop=appl[appl.AMT_INCOME_TOTAL.values>20000000]
appl_ti_drop


# In[818]:


appl.shape


# In[819]:


# Dropping this 1 row
appl=appl[~(appl.AMT_INCOME_TOTAL>2*10**7)]


# In[820]:


appl.shape


# ## Analysis of Categorical Variables

# ###### 1. Analysis of  ORGANIZATION_TYPE Column

# In[711]:


#Checking Value Counts
appl["ORGANIZATION_TYPE"].value_counts(dropna=False)


# In[714]:


# Replacing XNA value with 'np.nan'
appl["ORGANIZATION_TYPE"]= np.where(appl["ORGANIZATION_TYPE"]=="XNA",np.nan,appl["ORGANIZATION_TYPE"])
appl["ORGANIZATION_TYPE"].value_counts(dropna=False)


# ##### 2. Analyses of NAME_FAMILY_STATUS Column

# In[715]:


# Checking the NAME_FAMILY_STATUS Column
appl.NAME_FAMILY_STATUS.value_counts()

We can clearly see that Married/Civil Marriage and Single/Not married are same so we can merge these using replace function
# In[79]:


appl['NAME_FAMILY_STATUS'].replace({'Civil marriage': 'Married','Single/not married': 'Single' },inplace=True)


# In[80]:


appl.NAME_FAMILY_STATUS.value_counts()


# In[81]:


# Dropping the unknown rows
appl=appl[~(appl.NAME_FAMILY_STATUS=='Unknown')]
appl[(appl.NAME_FAMILY_STATUS=='Unknown')]


# In[82]:


#Checking again
appl.NAME_FAMILY_STATUS.value_counts()


# ##### 3. Analyses of NAME_CONTRACT_TYPE Column

# In[92]:


appl.NAME_CONTRACT_TYPE.value_counts()

This Column Looks good
# ##### 4. Analysing AMT_CREDIT Column

# In[93]:


appl.AMT_CREDIT.describe()


# In[821]:


#Checking the boxplot
plt.figure(figsize=[10,5])
sns.boxplot(x=appl.AMT_CREDIT)
plt.title("Analysis of AMT_CREDIT",fontsize=13)
plt.show()

There is a uniform distribution of data in outliers. Hence, we have to leave this column as it is
# ##### 5. Analyses of  NAME_TYPE_SUITE Column

# In[96]:


# Checking the NAME_TYPE_SUITE Column
appl.NAME_TYPE_SUITE.value_counts()

We can see that Other_A and Other_B can be clubbed together.
# In[945]:


#Clubbing Other_a and Other_B
appl['NAME_TYPE_SUITE'].replace({'Other_B': 'Other_A'},inplace=True)


# In[946]:


appl.NAME_TYPE_SUITE.value_counts()


# In[947]:


plt.figure(figsize=[10,5])
sns.histplot(x=appl.NAME_TYPE_SUITE)
plt.title("Analysis of NAME_TYPE_SUITE",fontsize=13)
plt.xticks(rotation=50)
plt.show()

This column Looks good now
# ##### 6. Analysing NAME_INCOME_TYPE Column

# In[100]:


# Checking the NAME_INCOME_TYPE Column
appl.NAME_INCOME_TYPE.value_counts()


# In[101]:


appl.NAME_INCOME_TYPE.isnull().sum()

This column looks good. Hence, no correction needed
# ##### 7. Analysing NAME_EDUCATION_TYPE Column

# In[102]:


# Checking the NAME_INCOME_TYPE Column
appl.NAME_EDUCATION_TYPE.value_counts()


# In[103]:


#Correcting the names
appl['NAME_EDUCATION_TYPE'].replace({'Secondary / secondary special': 'Secondary'},inplace=True)


# In[104]:


appl.NAME_EDUCATION_TYPE.value_counts()


# ##### 8. Analyses of NAME_HOUSING_TYPE Column

# In[105]:


# Checking the NAME_HOUSING_TYPE Column
appl.NAME_HOUSING_TYPE.value_counts()

This column looks good. No changes needed
# ## Binning Variables for Anaysis of Data

# In[824]:


#Putting up AMT_INCOME_TOTAL in bins
appl['AMT_INCOME_TOTAL'].quantile([0,0.1,0.3,0.6,0.8,1])


# In[107]:


appl['INCOME']=pd.qcut(appl['AMT_INCOME_TOTAL'], q=[0,0.1,0.3,0.6,0.8,1], labels=['Very Low','Low','Medium','High','Very High'])


# In[149]:


appl['INCOME'].value_counts()


# In[109]:


#Putting up AGE in bins
appl['AGE'].quantile([0,0.2,0.4,0.6])


# In[110]:


appl['AGE_BIN']=pd.qcut(appl["AGE"],q=[0,0.2,0.4,0.60,0.8],labels=['0-20','20-40','40-60','60-80'])


# In[826]:


## Adding one more column that will be used for analysis later
appl['AMT_CREDIT'].quantile([0,0.1,0.3,0.6,0.8,1])

# Creating bins for Credit amount

bins1 = [0,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000,700000,750000,800000,850000,900000,1000000000]
slot = ['0-150000', '150000-200000','200000-250000', '250000-300000', '300000-350000', '350000-400000','400000-450000',
        '450000-500000','500000-550000','550000-600000','600000-650000','650000-700000','700000-750000','750000-800000',
        '800000-850000','850000-900000','900000-100000000000']

appl['AMT_CREDIT_RANGE']=pd.cut(appl['AMT_CREDIT'],bins=bins1,labels=slot)


# In[827]:


#Converting the column into float
appl['AMT_CREDIT_RANGE']=appl['AMT_CREDIT_RANGE'].value_counts().astype(float)


# In[404]:


appl.AMT_CREDIT_RANGE.shape


# # Data Analysis
# Dividing the dataset into two sub datasets of TARGET=1(Defaulter) and TARGET=0(Non Defaulter)
# In[406]:


appl_t0 = appl.loc[appl["TARGET"]==0]
appl_t1 = appl.loc[appl["TARGET"]==1]


# In[407]:


appl_t0.value_counts().sum()


# In[408]:


appl_t1.value_counts().sum()


# In[409]:


appl['TARGET'].value_counts(normalize=True)*100


# In[948]:


#Checking the imbalance percentage
imbalance=round(len(appl_t0)/len(appl_t1),2)
imbalance


# In[411]:


#Plotting the defaulters Vs Non defaulters
plt.pie(appl['TARGET'].value_counts(normalize=True)*100,labels=['Non Defaulter (TARGET=0)','Defaulter (TARGET=1)'],explode=(0,0.05),autopct='%1.f%%')
plt.title('TARGET Variable - DEFAULTER Vs NONDEFAULTER')
plt.show()

There is an imbalance between number of defaulter and non defaulters. More than 91% of people didn't default as opposed to 8% who defaulted.
# # Univariate Analysis (Categorical Variable)

# ##### 1. Analyses of INCOME Column

# In[830]:


#Univariate Analyses of INCOME Column with respect to Defaulters and Non Defaulters
plt.figure(figsize=(19,7))
plt.subplot(1,2,1)
appl_t0.INCOME.value_counts().plot(kind='bar',title="Income Distibution of Applications with Non Defaulters",color='pink')
plt.xticks(rotation=0)
plt.legend()
plt.subplot(1,2,2)
appl_t1.INCOME.value_counts().plot(kind='bar',title="Income Distibution of Applications with Defaulters ",color='pink')
plt.xticks(rotation=0)
plt.legend()
plt.show()

Inference: 
1. Medium Salary people in both defaulters and non defaulters are highest.
2. Very low salary people do not take much loans from the bank.
# ##### 2. Analysis of AGE_BIN Column

# In[835]:


#Univariate Analyses of AGE_BIN Column with respect to Defaulters and Non Defaulters
plt.figure(figsize=(19,7))
plt.subplot(1,2,1)
appl_t0.AGE_BIN.value_counts().plot(kind='bar',title="Age Distibution of Applications with Non Defaulters",color='orange')
plt.xticks(rotation=0)
plt.subplot(1,2,2)
appl_t1.AGE_BIN.value_counts().plot(kind='bar',title="Age Distibution of Applications with Defaulters",color='orange')
plt.xticks(rotation=0)
plt.show()

Inference:
1. Below 20 people has the highest count in defaulter and non defaulters. Hence, they are mostly students and have no source of income. They may take loans for studies or something else.
2. Intersting thi is that age group of 20-40 are lowest in Non defaulter category and 60-80 are lowest in defaulter category. hence, 60-80 people can be targeted by the bank for loans.
# ##### 3. Analyses of GENDER_CODE Column

# In[836]:


#Univariate Analyses of GENDER_CODE Column with respect to Defaulters and Non Defaulters
plt.figure(figsize=(19,7))
plt.subplot(1,2,1)
appl_t0.CODE_GENDER.value_counts().plot(kind='bar',title="Gender Distibution of Applications with Non Defaulters",color='red')
plt.xticks(rotation=0)
plt.subplot(1,2,2)
appl_t1.CODE_GENDER.value_counts().plot(kind='bar',title="Gender Distibution of Applications with Defaulters",color='red')
plt.xticks(rotation=0)
plt.show()

Inference:
1. Females are more in both the graph. They apply for loans more as compared to men.
# ##### 4. Analyses of OCCUPATION_TYPE Column

# In[838]:


#Univariate Analyses of OCCUPATION_TYPE Column with respect to Defaulters and Non Defaulters
plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
plt.title("OCCUPATION_TYPE Status of Applications with Non Defaulter")
appl_t0.OCCUPATION_TYPE.value_counts().plot(kind='bar',color='green')
plt.subplot(1,2,2)
plt.title("OCCUPATION_TYPE Status of Applications with Defaulter")
appl_t1.OCCUPATION_TYPE.value_counts().plot(kind='bar',color='green')
plt.show()

Inference:
1. Laborers,Sales Staff, Core Staff, Managers are the highest and contributes to 50 % of the data in Defaulters and Non Defaulters.
# ##### 5. Analyses of NAME_TYPE_SUITE Column

# In[508]:


#Univariate Analyses of NAME_TYPE_SUITE Column with respect to Defaulters and Non Defaulters
plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
plt.title("Suite Status of Applications with Non Defaulter")
appl_t0.NAME_TYPE_SUITE.value_counts().plot(kind='bar',color='green')
plt.legend()
plt.subplot(1,2,2)
plt.title("Suite Status of Applications with Defaulters")
appl_t1.NAME_TYPE_SUITE.value_counts().plot(kind='bar',color='green')
plt.legend()
plt.show()

Inference:
Mostly, there are Loan Unaccompanied in both the data.
# ###### Function for Plotting the variables

# In[450]:


# function to count plot for categorical variables
def plot(var):

    plt.style.use('ggplot')
    sns.despine
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,7))
    
    sns.countplot(x=var, data=appl_t0,ax=ax1,palette='bright')
    ax1.set_ylabel('Total Counts')
    ax1.set_title(f'Distribution of {var} for Non defaulters',fontsize=20)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")
   
    
    # Adding the normalized percentage for easier comparision between defaulter and non-defaulter
    for p in ax1.patches:
        ax1.annotate('{:.1f}%'.format((p.get_height()/len(appl_t0))*100), (p.get_x()+0.1, p.get_height()+60))
        
    sns.countplot(x=var, data=appl_t1,ax=ax2,palette='bright')
    ax2.set_ylabel('Total Counts')
    ax2.set_title(f'Distribution of {var} for Defaulters',fontsize=20)  
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right")
    
    
    # Adding the normalized percentage for easier comparision between defaulter and non-defaulter
    for p in ax2.patches:
        ax2.annotate('{:.1f}%'.format((p.get_height()/len(appl_t1))*100), (p.get_x()+0.1, p.get_height()+50))
    
    plt.show()
    


# ##### 6. Analyses of NAME_INCOME_TYPE

# In[839]:


#Univariate Analyses of NAME_INCOME_TYPE Column with respect to Defaulters and Non Defaulters
plot('NAME_INCOME_TYPE')

Inference: 
1. Working professionals contribute to 50 % in Non defaulters and 61 % in defaulters.

# ##### 7. Analyses of NAME_FAMILY_STATUS Column 

# In[452]:


#Univariate Analyses of NAME_FAMILY_STATUS Column with respect to Defaulters and Non Defaulters
plot('NAME_FAMILY_STATUS')

Inference: 
1. Married people, 71 % have Defaulted for payments.
# ##### 8. Analyses of NAME_EDUCATION_TYPE Column

# In[841]:


#Univariate Analyses of NAME_EDUCATION_TYPE Column with respect to Defaulters and Non Defaulters
plot('NAME_EDUCATION_TYPE')

Inference:
1. Secondary Education people have applied for loans and further analysis iss needed for conclusion
# ##### 9. Analyses of WEEKDAY_APPR_PROCESS_START Column

# In[845]:


#Univariate Analyses of WEEKDAY_APPR_PROCESS_START Column with respect to Defaulters and Non Defaulters
plot('WEEKDAY_APPR_PROCESS_START')

Inference:
1. Tuesday records the highest number of applicants
2. Sunday is the lowest
# ##### 12. Analyses of ORGANIZATION_TYPE Column

# In[850]:


#Univariate Analyses of ORGANIZATION_TYPE Column with respect to Non Defaulters
plt.figure(figsize=[20,5])
appl_t0['ORGANIZATION_TYPE'].value_counts().plot.bar()
plt.xticks(rotation=90)
plt.title('Analysis of Organization Type with Non Defaulters')
plt.show()


# In[851]:


#Univariate Analyses of ORGANIZATION_TYPE Column with respect to Defaulters
plt.figure(figsize=[20,5])
appl_t1['ORGANIZATION_TYPE'].value_counts().plot.bar()
plt.xticks(rotation=90)
plt.title('Analysis of Organization Type with Non Defaulters')
plt.show()

Inference:
1. People from Business Entity type-3 are highest among loan applicants
# ##### 13. Analyses of NAME_CONTRACT_TYPE Column

# In[853]:


#Univariate Analyses of NAME_CONTRACT_TYPE Column with respect to Defaulters and Non Defaulters
plot('NAME_CONTRACT_TYPE')

Inference: 
# ## Univariate Analysis (Continous Variable)

# In[858]:


# function to dist plot for continuous variables
def plot_c(var):

    plt.style.use('ggplot')
    sns.despine
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
    
    sns.distplot(a=appl_t0[var],ax=ax1,hist=False)

    ax1.set_title(f'Distribution of {var} for Non Defaulters',fontsize=15)
            
    sns.distplot(a=appl_t1[var],ax=ax2,hist=False)
    ax2.set_title(f'Distribution of {var} for Defaulters',fontsize=15)    
        
    plt.show()


# ##### 1. Analyses of TIME_YEARS_EMPLOYED Column 

# In[859]:


#Univariate Analyses of TIME_YEARS_EMPLOYED Column with respect to Defaulters and Non Defaulters
plot_c('TIME_YEARS_EMPLOYED')

Inference:
People who have experienced bracket of 0-10 years are mostly applying for loans
# ##### 2. Analyses of AGE Column

# In[520]:


#Univariate Analyses of AGE Column with respect to Defaulters and Non Defaulters
plot_c('AGE')

Inference: 
1. This is an important finding that there is a sharp decline in Defaulters in age group 40-60.
2. The curve in Non Defaulter seemed to be less sharp.
# ##### 3. Analyses of CNT_FAM_MEMBERS Column

# In[861]:


#Univariate Analyses of CNT_FAM_MEMBERS Column with respect to Defaulters and Non Defaulters
plot_c('CNT_FAM_MEMBERS')


# ##### 4. Analyses of CNT_CHILDREN Column

# In[862]:


#Univariate Analyses of CNT_CHILDREN Column with respect to Defaulters and Non Defaulters
plot_c('CNT_CHILDREN')


# ##### 5. Analyses of AMT_CREDIT Column

# In[863]:


#Univariate Analyses of AMT_CREDIT Column with respect to Defaulters and Non Defaulters
plot_c('AMT_CREDIT')


# ## Bivariate Analysis (Continous Vs Continous Variables)

# #####  Plotting AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AGE, TIME_YEARS_EMPLOYED columns

# In[899]:


#Pair Plot of applicant data with Non Defaulters
sns.pairplot(appl_t0[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AGE',"TIME_YEARS_EMPLOYED"]].fillna(0))
plt.show()

Inference: 
1. Amount Credit and Amount Annuity shows a strong relation between each other.
2. Amount Annuity decreases with the number of years of experience.
3. Amount Annuity becomes constant after a certain age.

# In[898]:


#Pair Plot of applicant data with Defaulters
sns.pairplot(appl_t1[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AGE',"TIME_YEARS_EMPLOYED"]].fillna(0))
plt.show()

Inference:
Inference: 
1. Amount Credit and Amount Annuity shows a strong relation between each other.
2. Amount Annuity decreases with the number of years of experience.
3. Amount Credit is decreased as the age and expereice increases and tend to be higher in the age group of 20-40 years


# ### (Continous Vs Categorical Variables)

# ##### 1. Plotting NAME_EDUCATION_TYPE vs AMT_CREDIT Column

# In[889]:


#ploting NAME_EDUCATION_TYPE vs AMT_CREDIT for each family status for Non Defaulters

sns.catplot(data =appl_t0, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',height=6,aspect=4, kind="boxen", palette="muted")
plt.title('Credit Amount vs Education Status for Defaulters')
plt.show()


# In[890]:


#ploting NAME_EDUCATION_TYPE vs AMT_CREDIT for each family status for Non Defaulters
sns.catplot(data =appl_t1, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient="v",height=6,aspect=4, kind="boxen", palette="muted")
plt.title('Credit Amount vs Education Status for Defaulters')
plt.show()

Inference: Points to be drawn from the graph above for (Non-Defaulters).
1. Customers with an academic degree have a higher credit limit, with c married being the most common.
2. Customers with less education have lower credit limits, with widows having the lowest.
3. Married clients have a larger credit amount in practically all schooling segments except lower secondary and academic degrees.

Conclusions to be drawn from the preceding graph for (Defaulters).
1. Customers with a married academic degree have a greater credit limit and, as a result, a higher default rate.
2. Married customers have larger credit amounts across all academic segments.
3. Customers with a lesser eductation have a lower credit limit.
4. The only two family types represented in academic degrees are single and married.

# ##### 2. plotting AMT_CREDIT prev vs NAME_HOUSING_TYPE Column

# In[894]:


# Box plotting for Credit amount prev vs Housing type in logarithmic scale for Non Defaulters
sns.catplot(x="NAME_HOUSING_TYPE", y="AMT_CREDIT", hue="TARGET", data=appl_t0, kind="violin",height=6,aspect=4,palette='husl')
plt.title('Prev Credit amount vs Housing type')
plt.show()


# In[895]:


# Box plotting for Credit amount prev vs Housing type in logarithmic scale for Defaulters
sns.catplot(x="NAME_HOUSING_TYPE", y="AMT_CREDIT", hue="TARGET", data=appl_t1, kind="violin",height=6,aspect=4,palette='husl')
plt.title('Prev Credit amount vs Housing type')
plt.show()

Inference:
 Points to be drawn from the graph above for (Non-Defaulters).
 1. People who belongs office apartments tend to have higher cresit limit followed by house apartments with Co-op apartment as the lowest
  Points to be drawn from the graph above for (Non-Defaulters).
 2. People from house apartments most likely to be defaulted.
 
# ### (Categorical Vs Categorical Variables)

# In[936]:


# Box plotting for NAME_INCOME_TYPE vs CODE_GENDER
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.title('Non defaulter')
sns.countplot(x='NAME_INCOME_TYPE',hue='CODE_GENDER',data=appl_t0)
plt.xticks(rotation=45)

plt.subplot(1,2,2)
plt.title('Defaulter')
sns.countplot(x='NAME_INCOME_TYPE',hue='CODE_GENDER',data=appl_t1)
plt.xticks(rotation=45)
plt.show()

Inference:
1. Females in all the categories are the highets.
# In[916]:


# Box plotting for NAME_INCOME_TYPE vs NAME_EDUCATION_TYPE
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.title('Non defaulter')
sns.countplot(x='NAME_INCOME_TYPE',hue='NAME_EDUCATION_TYPE',data=appl_t0)
plt.xticks(rotation=45)

plt.subplot(1,2,2)
plt.title('Defaulter')
sns.countplot(hue='NAME_EDUCATION_TYPE',x='NAME_INCOME_TYPE',data=appl_t1)
plt.xticks(rotation=45)
plt.show()


# In[925]:


# Box plotting for OCCUPATION_TYPE vs FLAG_OWN_REALTY
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.title('Non defaulter')
sns.countplot(x='OCCUPATION_TYPE',hue='FLAG_OWN_REALTY',data=appl_t0)
plt.xticks(rotation=90)

plt.subplot(1,2,2)
plt.title('Defaulter')
sns.countplot(x='OCCUPATION_TYPE',hue='FLAG_OWN_REALTY',data=appl_t1)
plt.xticks(rotation=90)
plt.show()


# In[ ]:





# In[931]:


# Box plotting for AME_HOUSING_TYPE vs FLAG_OWN_CAR
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.title('Non defaulter')
sns.countplot(x='NAME_HOUSING_TYPE',hue='FLAG_OWN_CAR',data=appl_t0)
plt.xticks(rotation=90)

plt.subplot(1,2,2)
plt.title('Defaulter')
sns.countplot(x='NAME_HOUSING_TYPE',hue='FLAG_OWN_CAR',data=appl_t1)
plt.xticks(rotation=90)
plt.show()


# In[ ]:





# ## Multivariate Analysis

# In[937]:


#Multivariate Analysis of Continous Columns
appl_con_bi=appl_t0[['SK_ID_CURR','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
                               'AMT_GOODS_PRICE','DAYS_BIRTH','DAYS_EMPLOYED','CNT_FAM_MEMBERS','REGION_RATING_CLIENT',
                              'REGION_POPULATION_RELATIVE','DAYS_ID_PUBLISH']]
plt.figure(figsize=(25,25))
sns.heatmap(appl_con_bi.corr(), fmt='.1f', cmap="Blues", annot=True)
plt.title("Correlation Matrix for Non-Defaulters",fontsize=30, pad=20 )
plt.show()


# In[866]:


#Multivariate Analysis of Continous Columns
appl_con_bi=appl_t1[['SK_ID_CURR','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
                               'AMT_GOODS_PRICE','DAYS_BIRTH','DAYS_EMPLOYED','CNT_FAM_MEMBERS','REGION_RATING_CLIENT',
                              'REGION_POPULATION_RELATIVE','DAYS_ID_PUBLISH']]
plt.figure(figsize=(25,25))
sns.heatmap(appl_con_bi.corr(), fmt='.1f', cmap="Blues", annot=True)
plt.title("Correlation Matrix for Defaulters",fontsize=30, pad=20 )
plt.show()

Inference:
The credit amount is inversely related to the date of birth, with the credit amount increasing as the age decreases.
The credit amount is inversely related to the number of children a client has, with the credit amount increasing as the number of children decreases.
Credit amount is inversely related to days worked, i.e. credit amount is larger for those who have recently begun working.
The amount of income a customer earns is inversely related to the number of children they have, i.e., those with fewer children earn more.
# # Segment 2: Previous Application Data

# In[528]:


prev.head()


# In[547]:


prev.info()


# In[563]:


# After dropping the column with more than 40 % of missing values
prev_null_v=(prev.isnull().sum()/len(prev)*100).sort_values(ascending=False)
prev_null_v


# In[566]:


prev.AMT_GOODS_PRICE.value_counts()


# In[572]:


#Checking Outliers
plt.figure(figsize=(6,7))
sns.boxplot(y=prev['AMT_GOODS_PRICE'])
plt.title("AMT_GOODS_PRICE",fontsize=13)
plt.show()


# In[571]:


prev.AMT_GOODS_PRICE.describe()


# In[574]:


#Filling up the null values with Median
prev_m_drop=prev.AMT_GOODS_PRICE.median()
prev.AMT_GOODS_PRICE.fillna(value=prev_m_drop,inplace=True)


# In[575]:


prev.AMT_GOODS_PRICE.isnull().sum()


# In[ ]:





# In[576]:


prev.AMT_ANNUITY.value_counts() 


# In[578]:


#Checking Outliers
plt.figure(figsize=(6,7))
sns.boxplot(y=prev['AMT_ANNUITY'])
plt.title("AMT_ANNUITY",fontsize=13)
plt.show()


# In[580]:


prev['AMT_ANNUITY'].describe()


# In[585]:


prev_q_drop=prev.AMT_ANNUITY.median()
prev.AMT_ANNUITY.fillna(value=prev_q_drop,inplace=True)


# In[ ]:





# In[ ]:





# In[586]:


prev.CNT_PAYMENT.isnull().sum()


# In[587]:


#Checking Outliers
plt.figure(figsize=(6,7))
sns.boxplot(y=prev['CNT_PAYMENT'])
plt.title("CNT_PAYMENT",fontsize=13)
plt.show()


# In[588]:


prev.CNT_PAYMENT.describe()


# In[591]:


prev_r_drop=prev.CNT_PAYMENT.median()
prev.CNT_PAYMENT.fillna(value=prev_r_drop,inplace=True)


# In[600]:


prev.isnull().sum()


# In[593]:


prev.head()


# ## Merging Database
# 

# In[606]:


combined_appl=appl.merge(prev,how='inner',on='SK_ID_CURR')


# In[609]:


combined_appl.info()


# In[614]:


#Contract Status of previous applications
plt.figure(figsize=(10,5))
combined_appl.NAME_CONTRACT_STATUS.value_counts().plot(kind='bar')
plt.show()

Inference:
# In[935]:


# Gender wise  Contract status count
plt.figure(figsize=(10,5))
df=combined_appl.pivot_table(index="NAME_CONTRACT_STATUS",columns="CODE_GENDER",values="SK_ID_CURR",aggfunc="count")
plt.title("Gender wise Contract status Count")
sns.heatmap(df,cmap="Greens")
plt.show()

Inference:
# In[616]:


# Income type wise Contract status count
plt.figure(figsize=(20,8))
df=combined_appl.pivot_table(index="NAME_CONTRACT_STATUS",columns="NAME_INCOME_TYPE",values="SK_ID_CURR",aggfunc="count")
plt.title("Income Type wise Contract status Count")
sns.heatmap(df,cmap="Reds")
plt.show()

Inference:
# In[940]:


#Orginzation wise Contract Status count
plt.figure(figsize=(20,8))
df=combined_appl.pivot_table(index="NAME_CONTRACT_STATUS",columns="ORGANIZATION_TYPE",values="SK_ID_CURR",aggfunc="count")
plt.title("Organization wise Contract status Count")
sns.heatmap(df,cmap="Purples")
plt.show()

Inference:
# In[623]:


#Education type wise Contract Status count
plt.figure(figsize=(20,8))
df=combined_appl.pivot_table(index="NAME_CONTRACT_STATUS",columns="NAME_EDUCATION_TYPE",values="SK_ID_CURR",aggfunc="count")
plt.title("Education type wise Contract status Count")
sns.heatmap(df,cmap="Reds")
plt.show()

Inference: 
# In[525]:


# function to count plot for categorical variables
def plot_uvi(var):

    plt.style.use('ggplot')
    sns.despine
    fig,ax = plt.subplots(1,1,figsize=(15,5))
    
    sns.countplot(x=var, data=prev,ax=ax,hue='NAME_CONTRACT_STATUS')
    ax.set_ylabel('Total Counts')
    ax.set_title(f'Distribution of {var}',fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    
    plt.show()


# In[625]:


plot_uvi('NAME_GOODS_CATEGORY')

Inference:
# In[527]:


plot_uvi('NAME_CLIENT_TYPE')

Inference:
# In[629]:


plot_uvi('PRODUCT_COMBINATION')

Inference:
# In[624]:


# % of Payment Difficulty - Education Type Vs Previous Loan Purpose
plt.figure(figsize=(20,8))
df=combined_appl.pivot_table(index="NAME_EDUCATION_TYPE",columns="NAME_CASH_LOAN_PURPOSE",values="TARGET",aggfunc=np.mean)
plt.title("% of Payment Difficulty - Education Type Vs Previous Loan Purpose")
sns.heatmap(df,cmap="Greens")
plt.show()

Conclusions:
- Instead of offering loans to co-op apartment housing, banks should focus on 'House apartment' and 'With parent' housing.
- People with a 'working' income category are more likely to have irregular or failed payments, according to the bank.
- In 'NAME CONTRACT TYPE,' the number of 'Revolving Loans' is quite low, and it also has the highest percentage of payment difficulties—around 10%. As a result, clients whose contract type in the prior application was 'Revolving loans' are the driving forces for Loan Defaulters.
# In[949]:


jovian.commit()


# In[ ]:





# In[ ]:




