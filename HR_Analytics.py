
# coding: utf-8

# # HR Analaytics - Attition Prediction

# In[4]:


#Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go
import plotly.offline as py


# In[5]:


#Import the raw dataset
raw_df = pd.read_csv("D:\Sandeep- DSDJ\python docs\WA_Fn-UseC_-HR-Employee-Attrition.csv")


# ## Begin Exploratory Data Analysis

# In[6]:


# Let us know look at the few data columns
print(raw_df.head(10))


# In[7]:


# Now, let us see the data_types of our columns
print(raw_df.describe())


# In[8]:


# Now we have to see if we have "null" valued columns
print(raw_df.info())


# In[9]:


# Dataquality check
print( raw_df.isnull().any())
clean_df = raw_df.drop_duplicates(subset='EmployeeNumber')


# In[10]:


# Check for employee count
print(clean_df.shape)


# In[11]:


# Calculating the attrition rate
attrition_rate = clean_df.Attrition.value_counts() / 1470
print("The attrition % is:", attrition_rate['Yes'] * 100)


# In[12]:


# Let do some Data Visualizations to get better understanding of our data
#Visualizing feature set
fig, ax = plt.subplots(5, 2, figsize=(9, 9))
sns.distplot(clean_df['TotalWorkingYears'], ax=ax[0, 0])
sns.distplot(clean_df['MonthlyIncome'], ax=ax[0, 1])
sns.distplot(clean_df['YearsAtCompany'], ax=ax[1, 0])
sns.distplot(clean_df['DistanceFromHome'], ax=ax[1, 1])
sns.distplot(clean_df['YearsInCurrentRole'], ax=ax[2, 0])
sns.distplot(clean_df['YearsWithCurrManager'], ax=ax[2, 1])
sns.distplot(clean_df['YearsSinceLastPromotion'], ax=ax[3, 0])
sns.distplot(clean_df['PercentSalaryHike'], ax=ax[3, 1])
sns.distplot(clean_df['YearsSinceLastPromotion'], ax=ax[4, 0])
sns.distplot(clean_df['TrainingTimesLastYear'], ax=ax[4, 1])
plt.tight_layout()
plt.show()


# In[13]:


#Compare a few feature values to see the relationships
fig, ax = plt.subplots(2, 2, figsize=(10, 10))  # 'ax' has references to all the four axes
sns.boxplot(clean_df['Attrition'], clean_df['MonthlyIncome'], ax=ax[0, 0])  # Plot on 1st axes
sns.boxplot(clean_df['Gender'], clean_df['MonthlyIncome'], ax=ax[0, 1])  # Plot on IInd axes
plt.xticks(rotation=90)
sns.boxplot(clean_df['Department'], clean_df['MonthlyIncome'], ax=ax[1, 0])  # Plot on IIIrd axes
plt.xticks(rotation=90)

sns.boxplot(clean_df['JobRole'], clean_df['MonthlyIncome'], ax=ax[1, 1])  # Plot on IV the axes
plt.show()

continuous = ['Attrition', 'Age', 'MonthlyIncome', 'JobLevel', 'TotalWorkingYears', 'PercentSalaryHike',
              'PerformanceRating']


# In[15]:


#Visualizing numerical variables with attrition
sns.pairplot(clean_df[continuous],  kind="reg", diag_kind = "kde"  , hue = 'Attrition', palette="husl" )
plt.show()


# In[21]:


#Visualizing a correlation matrix
corr = clean_df.corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap='coolwarm',
    square=True
)

plt.show()


# ## Building a Model

# In[24]:


# Let us convert the attrition (target variable) to numeric
# Define a dictionary for the target label
label = {'Yes': 1, 'No': 0}

# Use the pandas apply method to numerically encode our attrition target variable
clean_df["Attrition_num"] = clean_df["Attrition"].apply(lambda x: label[x])
target_df = clean_df["Attrition_num"]


# In[25]:


# Let us seperate our categorical data
categorical = []
for col, value in clean_df.iteritems():
    if value.dtype == 'object':
        categorical.append(col)
        categorical_df = clean_df[categorical]
categorical_df = categorical_df.drop('Attrition', axis=1) #Drop attrition as it is our target variable
categorical_df = pd.get_dummies(categorical_df) #onehot encoding categorical variables
print(categorical_df.info())


# In[26]:


# Seperating our numerical variables
numerical = clean_df.columns.difference(categorical)
numerical_df = clean_df[numerical]
numerical_df = numerical_df.drop('Attrition_num', axis=1)
print( numerical_df.info())


# In[27]:


# Let us now create our final_df
final_df = pd.concat([categorical_df, numerical_df], axis=1)


# In[ ]:


#Let us now visualize our target df
sns.countplot(target_df)
plt.show()


# We can clearly spot a data imbalance

# In[30]:


# Split feature and target df into train,test data sets
X_train, X_Test, y_train, Y_Test = train_test_split(final_df, target_df, test_size=0.2, random_state=42)
# As we can see we have an imbalance data set, we need to do sampling inorder to overcome this


# In[31]:


# As we can see we have an imbalance data set, we need to do sampling inorder to overcome this

# Upsample using SMOTE to treat our imbalance dataset
sm = SMOTE(random_state=12)
x_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)


# Let us train three models and see which model performs the best
# Fit the models to the Upsampled data

# In[35]:


#LOGISTIC_REGRESSION
lr = LogisticRegression()

# Fit the model to the Upsampling data
lr = lr.fit(x_train_sm, y_train_sm)

print ("\n\n ---Logistic Regression Model---")
lr_auc = roc_auc_score(Y_Test, lr.predict(X_Test))

print ("Logistic Regression AUC = %2.2f" % lr_auc)

lr2 = lr.fit(x_train_sm, y_train_sm)
print(classification_report(Y_Test, lr.predict(X_Test)))


# In[36]:


# Random Forest Model
rf = RandomForestClassifier()

#Applying 5-vold cross validation
rf_result = cross_val_score(rf, x_train_sm, y_train_sm, cv=5, scoring='f1')

rf_result.mean()


# In[38]:


rf = rf.fit(x_train_sm, y_train_sm)

print ("\n\n ---Random Forest Model---")
rf_roc_auc = roc_auc_score(Y_Test, rf.predict(X_Test))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(Y_Test, rf.predict(X_Test)))


# In[39]:


#Gradient Boosting
gbc = GradientBoostingClassifier()  

gbc = gbc.fit(x_train_sm,y_train_sm)


# In[40]:


#Applying 5-fold crossvalidation
gbc_result = cross_val_score(gbc, x_train_sm, y_train_sm, cv=5, scoring='f1')
gbc_result.mean()


# In[42]:


print ("\n\n ---Gradient Boosting Model---")
gbc_auc = roc_auc_score(Y_Test, gbc.predict(X_Test))
print ("Gradient Boosting Classifier AUC = %2.2f" % gbc_auc)
print(classification_report(Y_Test, gbc.predict(X_Test)))


# I would go with Random Forest here as it provides better generalization

# In[43]:


# Create ROC Graph
from sklearn.metrics import roc_curve
lr_fpr, lr_tpr, thresholds = roc_curve(Y_Test, lr.predict_proba(X_Test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(Y_Test, rf.predict_proba(X_Test)[:,1])
gbc_fpr, gbc_tpr, gbc_thresholds = roc_curve(Y_Test, gbc.predict_proba(X_Test)[:,1])


plt.figure()

# Plot Logistic Regression ROC
plt.plot(lr_fpr, lr_tpr, label='Logistic Regression (area = %0.2f)' % lr_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier (area = %0.2f)' % rf_roc_auc)

# Plot Decision Tree ROC
plt.plot(gbc_fpr, gbc_tpr, label='Gradient Boosting Classifier (area = %0.2f)' % gbc_auc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()


# In[57]:


# Confusion Matrix for Linear Regresion
confusion_matrix(Y_Test, lr.predict(X_Test))


# In[58]:


# Confusion Matrix for Random Forest
confusion_matrix(Y_Test, rf.predict(X_Test))


# In[59]:


# Confusion Matrix for Gradient Boosting
confusion_matrix(Y_Test, gbc.predict(X_Test))


# It depends on how much cost/weight you want on your two types of errors: (1) False Positives or (2) False Negatives
# 
# What's the coset of having a FN and FP?
# 
# Optimize Recall When:
# 
# You want to limit false negatives
# You will get more False Positives
# FN > FP
# Example: Predicting Cancer Patients or Credit Card Fraud
# 
# Optimize Precision When:
# You want to limit false positives
# FP > FN
# Example: Spam VS Not Spam

# In[53]:



feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = x_train_sm.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_importances = feature_importances.reset_index()
feature_importances


# Conclusion:
# Based on how we want to optimize recall or precision, management level decisions will be made. Management decisions can be made to optimize features like employee work life balance, overtime etc to decrease the attrition rate. Implementing Yoga or Stress Management courses can also help achieve job saticifaction.
