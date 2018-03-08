from sklearn import preprocessing 
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
submit=pd.read_csv("sampleSubmission.csv")
#print(train.info())
#print(train.describe())
dataTrain=train
#dataTrain=data[['
dataTest=test
rf=RandomForestClassifier(max_features='auto',criterion='gini',n_estimators=1500,min_samples_split=80,min_samples_leaf=2,oob_score=True,random_state=1,n_jobs=-1)

rf.fit(dataTrain.iloc[:,1:],dataTrain.iloc[:,0])
print("%.4f"%rf.oob_score_)
rf_res=rf.predict(dataTest)
submit['Action']=rf_res
submit['Action']=submit['Action'].astype(int)
submit.tocsv('answer.csv',index=False)
