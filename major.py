#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Getting the dataset
df=pd.read_csv('Mental Health Prediction Analysis.csv')


df=df.fillna(df.mode().iloc[0]) 



df=df.drop(columns=['Timestamp','What kind of behavior do you observe when you are depressed?',
                    'If yes, how did you overcome these depressed feelings?',
                    'If yes, what is your physical health?'],axis=1)


df.rename(columns = {'Age Group ':'Age Group','Education pursuing currently':'Education',
                     'Have you ever gone through any mental health problems previously?':'Previous Mental Health Problems',
                     'Trouble falling or staying asleep, or sleeping too much?':'Sleep Problem',
                     'Feeling bad about yourself - or that you are a failure?':'Feeling bad',
                     'Do you have trouble relaxing your mind?':'Mind relaxation problem',
                     'Are you constantly feeling afraid that something awful might happen?':'Anticipatory anxiety',
                     'Thoughts that you would be better off dead, or of hurting yourself':'Sucidal thoughts',
                     }, inplace = True)

df.rename(columns={'Which of these factors affect you the most?':'Factors',
                   'Did you try to overcome these feelings? If yes, how?':'Overcome the feelings'},
          inplace=True)


#Data Encoding
df=df.astype('string')


df['Factors-Friends']=[1 if "Friends" in x else 0 for x in df['Factors']]

df['Factors-Family']=[1 if "Family" in x else 0 for x in df['Factors']]
df['Factors-Education']=[1 if "Education" in x else 0 for x in df['Factors']]
df['Factors-Strangers']=[1 if "Strangers" in x else 0 for x in df['Factors']]
df['Factors-Health']=[1 if "Health" in x else 0 for x in df['Factors']]
df['Factors-Relatives']=[1 if "Relatives" in x else 0 for x in df['Factors']]
df['Factors-Career']=[1 if "Career" in x else 0 for x in df['Factors']]
df['Factors-Lover']=[1 if "Lover" in x else 0 for x in df['Factors']]
df['Factors-Others']=[1 if "None" in x else 0 for x in df['Factors']]
df['Factors-Marriage']=[1 if "Marriage" in x else 0 for x in df['Factors']]
df['Factors-Failure']=[1 if "Failure" in x else 0 for x in df['Factors']]

df=df.drop(columns='Factors',axis=1)




df['Family size'].replace(["SMALL","Small","Neuclear family (4 members)",
                           "4 members ","3-5"],"4",inplace=True)


overcome={"Yes":1,"No":0}
df['Overcome the feelings']=df['Overcome the feelings'].map(overcome)






#Encoding data
from sklearn.preprocessing import LabelEncoder
labelDict = {}
for feature in df:
    le = LabelEncoder()
    le.fit(df[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    df[feature] = le.transform(df[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] =labelValue
    

df.drop(columns='Family size',axis=1,inplace=True)

from sklearn.model_selection import train_test_split
X=df.drop(columns=["Previous Mental Health Problems","Education","Overcome the feelings"],axis=1)
y=df["Previous Mental Health Problems"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
print(X.columns)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
lr=LogisticRegression()
lr.fit(X_train,y_train)
svm=SVC(kernel="poly")
svm.fit(X_train,y_train)
rfc=RandomForestClassifier(n_estimators=20,max_depth=8)
rfc.fit(X_train,y_train)
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
DT= DecisionTreeClassifier()
DT.fit(X_train,y_train)

y_pred1=lr.predict(X_test)
y_pred2=svm.predict(X_test)
y_pred3=rfc.predict(X_test)
y_pred4=knn.predict(X_test)
y_pred5=DT.predict(X_test)

from sklearn.metrics import accuracy_score,precision_score,recall_score
acc1=accuracy_score(y_test,y_pred1)
acc2=accuracy_score(y_test,y_pred2) 
acc3=accuracy_score(y_test,y_pred3)
acc4=accuracy_score(y_test,y_pred4)
acc5=accuracy_score(y_test,y_pred5)

pre1=precision_score(y_test,y_pred1)
pre2=precision_score(y_test,y_pred2)
pre3=precision_score(y_test,y_pred3)
pre4=precision_score(y_test,y_pred4)
pre5=precision_score(y_test,y_pred5)

re1=recall_score(y_test,y_pred1)
re2=recall_score(y_test,y_pred2)
re3=recall_score(y_test,y_pred3)
re4=recall_score(y_test,y_pred4)
re5=recall_score(y_test,y_pred5)

metrics_table={'Accuracy':[acc1,acc2,acc3,acc4,acc5],'Precision':[pre1,pre2,pre3,pre4,pre5],'Recall':[re1,re2,re3,re4,re5]}
mt=pd.DataFrame(metrics_table)
mt=mt.rename(index={0:'Logisitic Regression',1:'Naive Bayes',2:'SVM',3:'Decision Tree',4:'KNN',5:'DT'})

mt

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred2)
cm

import pickle

pickle.dump(knn,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))