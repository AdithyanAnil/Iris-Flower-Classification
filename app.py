import pandas as pd
df=pd.read_csv('dataset/Iris.csv')
df.drop(['Id'],axis=1,inplace=True)
import streamlit as st

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
le=LabelEncoder()
df['Species']=le.fit_transform(df['Species'])
x=df.iloc[:,:-1]
y=df.iloc[:,-1]

ss=StandardScaler()
x=ss.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

Gnb=GaussianNB()
Gnb.fit(x_train,y_train)
y_predict=Gnb.predict(x_test)

acc_score=(accuracy_score(y_test,y_predict))

sample=[[6.2,	3.4,	5.4,	2.3]]
sample=ss.transform(sample)
prediction=Gnb.predict(sample)

