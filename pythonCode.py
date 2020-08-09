# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:11:19 2020

@author: surpraka
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as statsModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn import metrics

churn = pd.read_csv(r"C:\Users\surpraka\Desktop\MachineLearning\UpGrad\Logistic Regresion\Telecom Churn\churn_data.csv")
customer = pd.read_csv(r"C:\Users\surpraka\Desktop\MachineLearning\UpGrad\Logistic Regresion\Telecom Churn\customer_data.csv")
internet = pd.read_csv(r"C:\Users\surpraka\Desktop\MachineLearning\UpGrad\Logistic Regresion\Telecom Churn\internet_data.csv")

#Combine all data

df_1 = pd.merge(churn,customer,how='inner',on='customerID')
main = pd.merge(df_1,internet,how='inner',on='customerID')


                        # 1. Data Prepration
# ---------------------------------------------------------------------------


varlist = ['PhoneService','PaperlessBilling','Churn','Partner','Dependents']
def binarymap(feature):
    return feature.map({'Yes':1,'No':0})
main[varlist] = main[varlist].apply(binarymap)

columns = ['InternetService','Contract','PaymentMethod','gender']
dummies = pd.get_dummies(main[columns],drop_first=True)
main = pd.concat([main,dummies],axis=1)

m1 = pd.get_dummies(main['MultipleLines'],prefix='MutlipleLines')
ml1 = m1.drop(['MutlipleLines_No phone service'],1)
main = pd.concat([main,ml1],axis=1)

os = pd.get_dummies(main['OnlineSecurity'],prefix='OnlineSecurity')
os1 = os.drop(['OnlineSecurity_No internet service'],1)
main = pd.concat([main,os1],axis=1)

ob = pd.get_dummies(main['OnlineBackup'],prefix='OnlineBackup')
ob1 = ob.drop(['OnlineBackup_No internet service'],1)
main = pd.concat([main,ob1],axis=1)

dp = pd.get_dummies(main['DeviceProtection'],prefix='DeviceProtection')
dp1 = dp.drop(['DeviceProtection_No internet service'],1)
main = pd.concat([main,dp1],axis=1)

ts = pd.get_dummies(main['TechSupport'],prefix='TechSupport')
ts1 = ts.drop(['TechSupport_No internet service'],1)
main = pd.concat([main,ts1],axis=1)

st = pd.get_dummies(main['StreamingTV'],prefix='StreamingTV')
st1 = st.drop(['StreamingTV_No internet service'],1)
main = pd.concat([main,st1],axis=1)

sm = pd.get_dummies(main['StreamingMovies'],prefix='StreamingMovies')
sm1 = sm.drop(['StreamingMovies_No internet service'],1)
main = pd.concat([main,sm1],axis=1)

main = main.drop(['Contract','PaymentMethod','gender','InternetService',
                  'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                  'StreamingMovies','MultipleLines'],1)

main['TotalCharges'] = pd.to_numeric(main['TotalCharges'],downcast='float')

    #  2. Handle outliers or missing values
num_main = main[['tenure','MonthlyCharges','TotalCharges','SeniorCitizen']]
print(num_main.describe(percentiles=[.25,.5,.75,.90,.95,.99]))

# print(main.isnull().sum())
# print(main.isnull().sum()/len(main.index)*100,2)
# main = main[~np.isnan(main['TotalCharges'])]

                   #  3. Split into train and test and Scaling of data
# ---------------------------------------------------------------------------

X = main.drop(['Churn','customerID'],axis=1)
y = main['Churn']

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=100)

scaler = StandardScaler()
x_train[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(x_train[['tenure','MonthlyCharges','TotalCharges']])

churn= (sum(main['Churn'])/len(main['Churn'].index))*100
print(churn)

                    #  4. Data Visualization
# ---------------------------------------------------------------------------

# Heatmap
plt.figure(figsize=[20,20])
sb.heatmap(x_train.corr(),annot=True)
plt.show()

#Drop features which are highly correlated
x_test = x_test.drop(['MutlipleLines_No','OnlineSecurity_No','OnlineBackup_No',
                      'DeviceProtection_No','StreamingTV_No','StreamingMovies_No',
                      'TechSupport_No'],1)
x_train = x_train.drop(['MutlipleLines_No','OnlineSecurity_No','OnlineBackup_No',
                      'DeviceProtection_No','StreamingTV_No','StreamingMovies_No',
                      'TechSupport_No'],1)

# Heatmap
plt.figure(figsize=[20,20])
sb.heatmap(x_train.corr(),annot=True)
plt.show()

                   # 5. Model Building and feature elimination
# ---------------------------------------------------------------------------

lm = LinearRegression()
lm.fit(x_train,y_train)
rfe = RFE(lm,15)
rfe = rfe.fit(x_train,y_train)
list(zip(x_train.columns,rfe.support_,rfe.ranking_))
cols = x_train.columns[rfe.support_]
print(cols)
print(x_train.columns[~rfe.support_])


# Model 1
x_train_sm = statsModel.add_constant(x_train[cols])
logm1 = statsModel.GLM(y_train,x_train_sm,family = statsModel.families.Binomial()).fit()
print(logm1.summary())

y_train_pred = logm1.predict(x_train_sm)
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred_final   = pd.DataFrame({'Churn' : y_train.values , 'churn_prob':y_train_pred})
y_train_pred_final['custID'] = y_train.index
y_train_pred_final['predicted_churn'] = y_train_pred_final.churn_prob.map(lambda x: 1 if(x > 0.5) else 0)
print(y_train_pred_final[:10])

# Confusion Matrix
confusion = metrics.confusion_matrix(y_train_pred_final.Churn,y_train_pred_final.predicted_churn)
print(confusion)
print(metrics.accuracy_score(y_train_pred_final.Churn,y_train_pred_final.predicted_churn))

#  Calculate VIF
vif = pd.DataFrame()
vif['features'] = x_train[cols].columns
vif['VIF'] = [variance_inflation_factor(x_train[cols].values,i) for i in range(x_train[cols].shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF',ascending = False)
print(vif)


# Model 2
cols = cols.drop('MonthlyCharges')
x_train_sm2 = statsModel.add_constant(x_train[cols])
logm2 = statsModel.GLM(y_train,x_train_sm2,family = statsModel.families.Binomial()).fit()
print(logm2.summary())

vif = pd.DataFrame()
vif['features'] = x_train[cols].columns
vif['VIF'] = [variance_inflation_factor(x_train[cols].values,i) for i in range(x_train[cols].shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF',ascending = False)
print(vif)

# Model 3
cols = cols.drop('TotalCharges')
x_train_sm3 = statsModel.add_constant(x_train[cols])
logm3 = statsModel.GLM(y_train,x_train_sm3,family = statsModel.families.Binomial()).fit()
print(logm3.summary())

vif = pd.DataFrame()
vif['features'] = x_train[cols].columns
vif['VIF'] = [variance_inflation_factor(x_train[cols].values,i) for i in range(x_train[cols].shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF',ascending = False)
print(vif)

y_train_pred2 = logm3.predict(x_train_sm3)
y_train_pred2 = y_train_pred2.values.reshape(-1)
y_train_pred_final2   = pd.DataFrame({'Churn' : y_train.values , 'churn_prob':y_train_pred})
y_train_pred_final2['custID'] = y_train.index
y_train_pred_final2['predicted_churn'] = y_train_pred_final2.churn_prob.map(lambda x: 1 if(x > 0.5) else 0)
    

                        # 6. Model Evaluation
# ---------------------------------------------------------------------------


# Confusion Matrix and Accuracy
confusion = metrics.confusion_matrix(y_train_pred_final2.Churn,y_train_pred_final2.predicted_churn)
print(confusion)
print(metrics.accuracy_score(y_train_pred_final2.Churn,y_train_pred_final2.predicted_churn))

# Sensitivity and Specificity
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[1,0]
FN = confusion[0,1]

sensitivity = TP/float(TP+FP)
specificity = TN/float(TN+FN)

print(sensitivity)
print(specificity)

# ROC Curve

# Defining the function to plot the ROC curve
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None

# Calling the function
draw_roc(y_train_pred_final2.Churn, y_train_pred_final2.churn_prob)


                    # 7. Finding the Optimal Threshold
# ---------------------------------------------------------------------------


cutt_off = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
num = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:
   predicted_churns = y_train_pred_final2.churn_prob.map(lambda x: 1 if(x > i) else 0)
   cm = metrics.confusion_matrix(y_train_pred_final2.Churn, predicted_churns)
   total = sum(sum(cm))
   
   accuracy = (cm[0,0]+cm[1,1])/total
   speci = cm[0,0]/(cm[0,0]+cm[0,1])
   sensi = cm[1,1]/(cm[1,0]+cm[1,1])
   
   cutt_off.loc[i] =[ i ,accuracy,sensi,speci]

cutt_off.plot.line(x='prob',y=['accuracy','sensi','speci'])
plt.show()

#  Using curve we choose 0.3 as a cutt off
predicted_churns = y_train_pred_final2.churn_prob.map(lambda x: 1 if(x > 0.3) else 0)
cm2 = metrics.confusion_matrix(y_train_pred_final2.Churn, predicted_churns)
print("Confusion Matrix is :")
print(cm2)
total = sum(sum(cm2))

finalAccuracy = (cm2[0,0]+cm2[1,1])/total
finalSpeci = cm2[0,0]/(cm2[0,0]+cm2[0,1])
finalSensi = cm2[1,1]/(cm2[1,0]+cm2[1,1])

print("Accuracy : "+str(finalAccuracy))
print("Speci : "+str(finalSpeci))
print("Sensi : "+str(finalSensi))