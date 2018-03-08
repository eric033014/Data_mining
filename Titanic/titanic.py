from sklearn import preprocessing 
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
rf = RandomForestClassifier(max_features='auto',criterion='gini',
                             n_estimators=1500,
                             min_samples_split=60,
                             min_samples_leaf=3,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)





rf.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
print("%.4f" % rf.oob_score_)
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
dataTrain=dataTrain.values
X = dataTrain[0::, 1::]
y = dataTrain[0::, 0]

acc_dict = {}


for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_ = [] # array of predictions from the classifiers
    rf.fit(X_train,y_train)
    pred=rf.predict(X_test)
    acc=accuracy_score(y_test,pred)
    print(acc)





rf_res =  rf.predict(dataTest)
submit['Survived'] = rf_res
#print(rf_res)
submit['Survived'] = submit['Survived'].astype(int)
submit.to_csv('submit.csv', index= False)

"""
def mix(x):
    n_methods=len(classifiers)
    y_mix=[np.multiply(x[i],y_[i])for i in range(n_methods)]
    y_mix=np.sum(y_mix,axis=0)
    y_mix=np.array([round(y).astype(int) for y in y_mix])
    acc=accuracy_score(y_test,y_mix)
    return -acc




import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(
            max_features='auto',criterion='gini',
                             n_estimators=1500,
                             min_samples_split=80,
                             min_samples_leaf=4,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1
        ),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
dataTrain=dataTrain.values
X = dataTrain[0::, 1::]
y = dataTrain[0::, 0]

acc_dict = {}

x0 = [1 / len(classifiers)]*len(classifiers)

total=[]
a=0

for clf in classifiers:
    name= clf.__class__.__name__
    clf.fit(X,y)
    train_predictions=clf.predict(dataTest)
    
    if a==0:
        total=train_predictions
        a=1
    else:
        for i in range (len(total)):
            total[i]=total[i]+train_predictions[i]
print(total)

for i in range(len(total)):
    if total[i] >=5:
        total[i]=1
    else:
        total[i]=0
print(total)
submit['Survived'] = total
submit['Survived'] = submit['Survived'].astype(int)
submit.to_csv('submit.csv', index= False)


total=[[]]
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_ = [] # array of predictions from the classifiers

    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        test_predictions= clf.predict(dataTest)
        #total= total +test_predictions
        #print("datatest: ",test_predictions)
        acc = accuracy_score(y_test, train_predictions)
        #print(name)
        y_.append(train_predictions) # prediction of current classifier appended
        #print(y_)
        if name in acc_dict:
            acc_dict[name] += acc

        else:
            acc_dict[name] = acc
    #print(y_[0])
    # Computing mixed model
    
    res = minimize(mix, x0, bounds=[(0, 1)] * len(classifiers))
    weights = res.x
    acc = -res.fun # best accuracy reached through mixing
    if 'mix' in acc_dict:
        acc_dict['mix'] += acc
    else:
        acc_dict['mix'] = acc

total=[]
a=0
for clf in classifiers:
    name= clf.__class__.__name__
    clf.fit(X,y)
    train_predictions=clf.predict(dataTest)
    
    if a==0:
        total=train_predictions
        a=1
    else:
        for i in range (len(total)):
            total[i]=total[i]+train_predictions[i]
print(total)

for i in range(len(total)):
    if total[i] >=5:
        total[i]=1
    else:
        total[i]=0
#print(total)

submit['Survived'] = total
submit['Survived'] = submit['Survived'].astype(int)
submit.to_csv('submit.csv', index= False)




for clf in acc_dict:
    print(clf," ",acc_dict[clf])
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.show()
"""
