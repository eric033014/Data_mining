from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold;
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
import xgboost as xgb
import re
#%matplotlib inline
pd.options.mode.chained_assignment = None
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submit = pd.read_csv('gender_submission.csv')
data = train.append(test)
data.reset_index(inplace=True, drop=True)
data['Family_Size']=data['Parch']+data['SibSp']
g=sns.FacetGrid(data,col='Survived')
#plt.show(g.map(sns.distplot,'Family_Size',kde=False))
data['Title1'] = data['Name'].str.split(", ", expand=True)[1]
data['Name'].str.split(", ", expand=True).head(3)
data['Title1'] = data['Title1'].str.split(".", expand=True)[0]
data['Title2'] = data['Title1'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],
         ['Miss','Mrs','Miss','Mr','Mr','Mrs','Mrs','Mr','Mr','Mr','Mr','Mr','Mr','Mrs'])
print(data.groupby(['Title2','Pclass'])['Age'].mean())
data['Ticket_info'] = data['Ticket'].apply(lambda x : x.replace(".","").replace("/","").strip().split(' ')[0] if not x.isdigit() else 'X')
data['Embarked'] = data['Embarked'].fillna('S')
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
data["Cabin"] = data['Cabin'].apply(lambda x : str(x)[0] if not pd.isnull(x) else 'NoCabin')
data['Sex'] = data['Sex'].astype('category').cat.codes
data['Embarked'] = data['Embarked'].astype('category').cat.codes
data['Pclass'] = data['Pclass'].astype('category').cat.codes
data['Title1'] = data['Title1'].astype('category').cat.codes
data['Title2'] = data['Title2'].astype('category').cat.codes
data['Cabin'] = data['Cabin'].astype('category').cat.codes
data['Ticket_info'] = data['Ticket_info'].astype('category').cat.codes
dataAgeNull = data[data["Age"].isnull()]
dataAgeNotNull = data[data["Age"].notnull()]
remove_outlier = dataAgeNotNull[(np.abs(dataAgeNotNull["Fare"]-dataAgeNotNull["Fare"].mean())>(4*dataAgeNotNull["Fare"].std()))|
                      (np.abs(dataAgeNotNull["Family_Size"]-dataAgeNotNull["Family_Size"].mean())>(4*dataAgeNotNull["Family_Size"].std()))
                     ]
rfModel_age = RandomForestRegressor(n_estimators=2000,random_state=42)
ageColumns = ['Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title1', 'Title2','Cabin','Ticket_info']
rfModel_age.fit(remove_outlier[ageColumns], remove_outlier["Age"])

ageNullValues = rfModel_age.predict(X= dataAgeNull[ageColumns])
dataAgeNull.loc[:,"Age"] = ageNullValues
data = dataAgeNull.append(dataAgeNotNull)
data.reset_index(inplace=True, drop=True)
dataTrain = data[pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])
dataTest = data[~pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])
dataTrain = dataTrain[['Survived', 'Age', 'Embarked', 'Fare',  'Pclass', 'Sex', 'Family_Size', 'Title2','Ticket_info','Cabin']]
dataTest = dataTest[['Age', 'Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title2','Ticket_info','Cabin']]
rf = RandomForestClassifier(
    criterion='gini',
    n_jobs=-1,
    n_estimators= 1500,
    warm_start=True,
     #'max_features': 0.2,
    oob_score=True,
    max_depth=6,
    min_samples_split=80,
    random_state=1,
    min_samples_leaf=2,
    max_features= 'sqrt',
    verbose= 0
)

# Extra Trees Parameters
et=ExtraTreesClassifier(
    n_jobs=-1,
    n_estimators=500,
    #'max_features': 0.5,
    max_depth=8,
    min_samples_leaf= 2,
    verbose= 0
)
# AdaBoost parameters
ada=AdaBoostClassifier(
    n_estimators=5000,
    learning_rate= 0.75
)

# Gradient Boosting parameters
gb= GradientBoostingClassifier(
    n_estimators= 500,
     #'max_features': 0.2,
    max_depth=5,
    min_samples_leaf= 2,
    verbose=0
)
re=LogisticRegression()
# Support Vector Classifier parameters
sv=SVC(
    kernel= 'linear',
    C = 0.025
    )
SEED=0
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self,x,y):
        return self.clf.fit(x,y)

    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)

"""def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
"""


# Create 5 objects that represent our 4 models
"""rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
"""
# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data

# Create our OOF train and test predictions. These base results will be used as new features

#rf.fit(dataTrain.iloc[:,1:],dataTrain.iloc[:,0])
#et.fit(dataTrain.iloc[:,1:],dataTrain.iloc[:,0])




#sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)







"""
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier
"""






re.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])


re_res=re.predict(dataTrain.iloc[:,1:])




rf.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
et.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
et_res=et.predict(dataTrain.iloc[:,1:])
rf_res=rf.predict(dataTrain.iloc[:,1:])
ada.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
ada_res=ada.predict(dataTrain.iloc[:,1:])
gb.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
gb_res=gb.predict(dataTrain.iloc[:,1:])
sv.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
sv_res=sv.predict(dataTrain.iloc[:,1:])
et_res =  et.predict(dataTest)
rf_res =  rf.predict(dataTest)
ada_res =  ada.predict(dataTest)
gb_res =  gb.predict(dataTest)
sv_res =  sv.predict(dataTest)

a=0
total=et_res
for i in range(len(ada_res)):
    if a==0:
        a=a+1
        total[i]=total[i]+rf_res[i]+ada_res[i]+gb_res[i]+sv_res[i]
    else:
        total[i]=total[i]+rf_res[i]+ada_res[i]+gb_res[i]+sv_res[i]

for i in range(len(ada_res)):
    if (total[i]/5)<0.5:
        total[i]=0
    else:
        total[i]=1

#print("ET: ",roc_auc_score(et_res,dataTrain.iloc[:,0:1]))
#print("RF: ",roc_auc_score(rf_res,dataTrain.iloc[:,0:1]))
#print("ADA: ",roc_auc_score(ada_res,dataTrain.iloc[:,0:1]))
#print("GB: ",roc_auc_score(gb_res,dataTrain.iloc[:,0:1]))
#print("SVC: ",roc_auc_score(sv_res,dataTrain.iloc[:,0:1]))
#print("RE: ",roc_auc_score(re_res,dataTrain.iloc[:,0:1]))




print("%.4f" % rf.oob_score_)

ada_res =  ada.predict(dataTest)
submit['Survived'] = total
submit['Survived'] = submit['Survived'].astype(int)
submit.to_csv('submit.csv', index= False)
