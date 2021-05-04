# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #plot
import matplotlib.pyplot as plt #plot
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]

#data import
train= pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')

#data exploration
test.info() #less a column called 'Survived' which is the target
train
train.info()
train.isnull().sum()
cor_col = train[['Pclass','SibSp','Parch','Age','Fare']]
cor_col_wo_age = cor_col.drop('Age',axis=1)

sns.heatmap(cor_col.corr())
plt.show()
from scipy import stats


train[train.Embarked.isnull()][['Survived','Pclass','Embarked']]
embarked = train.groupby(['Embarked','Survived'])['PassengerId'].count()

chi = pd.pivot_table(train, values= 'PassengerId', index='Survived', columns='Embarked',aggfunc='count')

stats.chi2_contingency(chi)
surviverate_f = train[train['Sex']=='female']['Survived'].sum()/len(train[train['Sex']=='female'])
surviverate_m = train[train['Sex']=='male']['Survived'].sum()/len(train[train['Sex']=='male'])
numerical = train[['Age','Fare']]
ordinal = train[['Parch','Pclass','SibSp']]
categorical = train[['Survived','Sex','Ticket','Cabin','Embarked']]
pd.pivot_table(train, index='Survived', values=['Age','Fare','Parch','Pclass','SibSp'], aggfunc=[np.median,np.mean])

#drawing graphs
for i in numerical.columns:
    plt.hist(numerical[i])
    plt.title(i)
    plt.show()
bar = pd.concat([ordinal,categorical])
for i in bar.columns:
    sns.barplot(bar[i].value_counts().index,bar[i].value_counts()).set_title(i)
    plt.show()
    
#create combined
train['train_data'] = 1
test['train_data'] = 0
test['Survived'] = np.NaN
all_data = pd.concat([train,test])
all_data.reset_index(inplace=True, drop=True)

#check
all_data[all_data.duplicated()]


#version2 complete start with the order of feature as follow
#Agelevel(use feature engineered title,
#use famsize, and other than that the rest of info from that should used for death_connected)
#Sex
#Pclass related(cabin, embarked, fare, they seem to be all correlated, so needed to be integrated)
#Connected Death(survive or die together)


#title feature engineer
all_data['Title'] = all_data['Name'].str.split(", ", expand=True)[1]
all_data['Title'] = all_data['Title'].str.split(".", expand=True)[0]
all_data['Title'].unique()
all_data['Title'] = all_data['Title'].replace('Mlle', 'Miss')
all_data['Title'] = all_data['Title'].replace('Ms', 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')
all_data['Title'] = all_data['Title'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data['Title'].value_counts()
all_data[(all_data['Age']<12) & (all_data['Sex'] == 'female')][['Title','Age']]
#little girl are called Miss
#master are used for only 12 below and male
#age 15 and below has much higher survival rate
g = sns.FacetGrid(all_data, col="Survived",  row="Sex")
g = g.map(plt.hist, "Age")
#seems no survived difference between sex for kids
###Decided to put age 15 and under into 1 and others into 0
###Use title to manually predict agelevel 1 or 0
###Master must be 1, Mr must be 0, Mrs must be 0
###fmsz=0 must be 0

#First and foremost
#predict agelevel
['Family_size'] = all_data['Parch'] + all_data['SibSp']
nullage = all_data[all_data.Age.isnull()]
nofam = nullage[nullage.Family_size == 0]
nofam['Title'].value_counts()

def Age_15 (row) : 
    if row['Age'] <= 15:
        return 1
    if row['Age'] > 15:
        return 0
    else:
        return None
all_data['Age_15'] = all_data.apply(lambda row: Age_15 (row), axis = 1)
import math
def Title (row):
    if math.isnan(row['Age_15']) == True:
        if row['Title'] == 'Master' :
            return 1
        if row['Title'] == 'Mrs' :
            return 0
    else:
        return row['Age_15']
all_data['Age_15'] = all_data.apply(lambda row: Title (row), axis = 1)
def Fam (row):
    if math.isnan(row['Age_15']) == True:
        if row['Family_size'] == 0:
            return 0
    else:
        return row['Age_15']
all_data['Age_15'] = all_data.apply(lambda row: Fam (row), axis = 1)
all_data[all_data['Age_15'].isnull() & (all_data['Parch'] != 0) & (all_data['Title'] == 'Miss')]
#only 38 undecided data
#where 21 mr can be ignored since the only litter chance can they fall in age 13 14 15
#only 10 miss come with parent but 8/9 died and another 1 is test data who is also a connected data with others
#so put all 38 into age_15 0
def Rest (row):
    if math.isnan(row['Age_15']) == True:
        return 0
    else:
        return row['Age_15']
all_data['Age_15'] = all_data.apply(lambda row: Rest (row), axis = 1)

#Connected Survival
all_data[['Ticket','Fare','Family_size','Cabin','Embarked','Pclass','Survived','Connection']].sort_values('Fare',ascending=False).head(50)
all_data[(all_data.Pclass==1) & (all_data.Cabin_Yes==0) & (all_data.Tic_num == 1)]
# Filling missing values
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())
sns.distplot(np.log(all_data[all_data['Survived']==0]['Fare']+1), hist=True, rug=False)
sns.distplot(np.log(all_data[all_data['Survived']==1]['Fare']+1), hist=True, rug=False)
all_data['Log_Fare'] = np.log(all_data['Fare']+1)
# Making Bins
all_data['FareBin_4'] = pd.cut(all_data['Log_Fare'], 4)
all_data['FareBin_5'] = pd.cut(all_data['Log_Fare'], 5)
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
all_data['FareBin_Code_4'] = label.fit_transform(all_data['FareBin_4'])
all_data['FareBin_Code_5'] = label.fit_transform(all_data['FareBin_5'])
all_data.groupby('FareBin_Code_4')['Survived'].count()
all_data.groupby('FareBin_Code_5')['Survived'].count()
plt.figure()
plt.hist([np.log(all_data[all_data['Survived']==1]['Fare']+1),np.log(all_data[all_data['Survived']==0]['Fare']+1)],bins=15,stacked=True)
plt.show()
#we get that ticket and fare are same
#same tickets are only shared by family and cabin
#same tickets tends to survive together
dict = all_data['Ticket'].value_counts().to_dict()
all_data['Tic_num'] = all_data['Ticket'].map(dict)
Repeat = all_data.drop(all_data[all_data['Tic_num']<2].index)
Repeat[Repeat['Family_size'] == 0].sort_values('Ticket')[['Survived','Ticket']]
#127 rows where fam=0 ticket owned by >1
#oh shit just discover that some kids are there not with fam but with others but thanks god not many
#not_fam still tend to survived together
#decide to make connections based on their tickets
all_data['Connection'] = 0.5
for tic, group in all_data.groupby('Ticket'):
    df = pd.DataFrame(group)
    if df.Tic_num.sum() > 1:
        con = df.Survived.max()
        list = df.PassengerId
        if con == 1:
            all_data.loc[all_data.PassengerId.isin(list),'Connection'] = 1
        if con == 0:
            all_data.loc[all_data.PassengerId.isin(list),'Connection'] = 0
all_data.sort_values('Ticket')[['Ticket','Family_size','Survived','Connection','Tic_num']].tail(30)
all_data.groupby('Connection')['Survived'].mean()
all_data[(all_data.Family_size > 0) & (all_data.Tic_num == 1) & (all_data.train_data == 0)].sort_values('Family_size',ascending=False).drop(['PassengerId','Survived','Ticket','Tic_num','Connection','Cabin_Yes','Size','Embarked_CQ','FareBin_4','FareBin_5','FareBin_Code_4','Log_Fare'],axis=1)
all_data[all_data.Family_size==6][['Survived','Name','Parch','SibSp','Connection','Tic_num']]
#connection = 1 survival rate = 0.79

###for people who have family but tic_num = 1
###adjust connection for them
all_data[(all_data.Tic_num == 1) & (all_data.Family_size > 0)][['Title','Ticket','Family_size','Survived','Connection','Tic_num']]
def Home (row):
    if (row.Tic_num == 1) & (row.Family_size > 0) & (row.train_data == 1):
        return row.Survived
    else:
        return row.Connection
all_data['Connection'] = all_data.apply(lambda row: Home (row), axis = 1)


#Ship related
#Cabin, Pclass, Fare, Embarked
all_data[all_data['Pclass'] == 1]['Embarked'].value_counts()
all_data[all_data['Pclass'] == 2]['Embarked'].value_counts()
all_data[all_data['Pclass'] == 3]['Embarked'].value_counts()
all_data.Cabin = all_data.Cabin.fillna('XXX')
def Cabin(row):
    if row['Cabin'] == 'XXX':
        return 0
    else:
        return 1
all_data['Cabin_Yes'] = all_data.apply(lambda row: Cabin(row), axis = 1)
pd.pivot_table(all_data,values='Survived', index='Cabin_Yes', aggfunc='mean')
pd.pivot_table(all_data,values='Survived', index=['Cabin_Yes','Pclass'],columns = 'Embarked', aggfunc='sum')
pd.pivot_table(all_data,values='Survived', index=['Cabin_Yes','Pclass'],columns = 'Embarked', aggfunc='mean')
pd.pivot_table(all_data,values='Survived', columns = 'Embarked', aggfunc='mean')
#Cabin seems like a more determinant factor than embark and pclass
plt.hist(all_data[all_data['Survived'] == 0]['Fare'])
plt.hist(all_data[all_data['Survived'] == 1]['Fare'])
#doesn't seem to cause difference inside Pclass

###for family size and embarked and Farecut
def Size(row):
    if (row['Family_size'] >= 2) & (row['Family_size'] <= 4):
        return 1
    else:
        return 0
all_data['Size'] =all_data.apply(lambda row: Size(row), axis = 1)
def Embark(row):
    if (row['Embarked'] == 'S'):
        return 0
    else:
        return 1
all_data['Embarked_CQ'] = all_data.apply(lambda row: Embark(row), axis = 1)
def FareRange(row):
    

###Necessary feature 1.Sex 2.Agelevel 3.Cabin_Yes 4.Connection
###Try 1.Pclass 2.Embarked 3.family_size 4.Fare
data = all_data.drop(['PassengerId','Name','Cabin','Parch','Ticket','Age','SibSp','Tic_num','Title','FareBin_4','FareBin_5','FareBin_Code_4','Log_Fare'],axis=1)
data.isnull().sum()
data = data.drop(['Fare','Family_size','Embarked','Size','Cabin_Yes','Embarked_CQ','Pclass'],axis=1)
### 1.Pclass 2.family_size seems to perform better

#preprocessing
#turn to numeric or one hot encode
#log fare for include fare
#standscale for other model
data_dummies = pd.get_dummies(data)
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
num = ['FareBin_Code_5']
data_dummies[num] = scale.fit_transform(data_dummies[num])
X_train = data_dummies[ data_dummies['train_data'] == 1 ].drop(['Survived','train_data'],axis=1)
y_train = data_dummies[ data_dummies['train_data'] == 1 ].Survived
X_test = data_dummies[ data_dummies['train_data'] == 0 ].drop(['Survived','train_data'],axis=1)



#model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
#rf
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train,y_train,cv=5)
print(cv)
print(cv.mean())
#Support Vector Machine
svc = SVC(probability = True)
cv = cross_val_score(svc,X_train,y_train,cv=5)
print(cv)
print(cv.mean())
#XGB
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state =1)
cv = cross_val_score(xgb,X_train,y_train,cv=5)
print(cv)
print(cv.mean())

#ensemble
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators = [('rf',rf),('svc',svc),('xgb',xgb)], voting = 'soft')
cv = cross_val_score(voting_clf,X_train,y_train,cv=5)
print(cv)
print(cv.mean())

#other model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
#Kneighbors
knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train,y_train,cv=5)
print(cv)
print(cv.mean())
#Gaussian Bayes
gnb = GaussianNB()
cv = cross_val_score(gnb,X_train,y_train,cv=5)
print(cv)
print(cv.mean())
#Logistic Regression
lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train,y_train,cv=5)
print(cv)
print(cv.mean())

#Decision Tree
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt,X_train,y_train,cv=5)
print(cv)
print(cv.mean())


#Model Tuning
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV 
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))
#knn
knn = KNeighborsClassifier()
param_grid = {'n_neighbors' : [3,5,7,9],
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree','kd_tree'],
              'p' : [1,2]}
clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_knn = clf_knn.fit(X_train,y_train)
clf_performance(best_clf_knn,'KNN')
#Svc
svc = SVC(probability = True)
param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],
                                  'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100, 1000]}]
clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_svc = clf_svc.fit(X_train,y_train)
clf_performance(best_clf_svc,'SVC')
#rf
rf = RandomForestClassifier(random_state = 1)
param_grid =  {'n_estimators': [400,450,500,550],
               'criterion':['gini','entropy'],
                                  'bootstrap': [True],
                                  'max_depth': [15, 20, 25],
                                  'max_features': ['auto','sqrt', 10],
                                  'min_samples_leaf': [2,3],
                                  'min_samples_split': [2,3]}   
clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_rf = clf_rf.fit(X_train,y_train)
clf_performance(best_clf_rf,'Random Forest')
best_rf = best_clf_rf.best_estimator_.fit(X_train,y_train)
feat_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh')
#xgb
xgb = XGBClassifier(random_state = 1)
param_grid = {
    'n_estimators': [450,500,550],
    'colsample_bytree': [0.75,0.8,0.85],
    'max_depth': [None],
    'reg_alpha': [1],
    'reg_lambda': [2, 5, 10],
    'subsample': [0.55, 0.6, .65],
    'learning_rate':[0.5],
    'gamma':[.5,1,2],
    'min_child_weight':[0.01],
    'sampling_method': ['uniform']
}
clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(X_train,y_train)
clf_performance(best_clf_xgb,'XGB')

#ensembel predict and submit
best_knn = best_clf_knn.best_estimator_
best_svc = best_clf_svc.best_estimator_
best_rf = best_clf_rf.best_estimator_
best_xgb = best_clf_xgb.best_estimator_
voting_clf_xgb = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('xgb', best_xgb)], voting = 'soft')
print('voting_clf_xgb :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=5))
print('voting_clf_xgb mean :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=5).mean())
voting_clf_xgb.fit(X_train, y_train)

#Predict
best_rf = best_clf_rf.best_estimator_
rf.fit(X_train, y_train)
prediction = best_rf.predict(X_test)
answer = pd.read_csv('../input/answer/submit.csv') #79
(prediction != answer['Survived']).sum()


#submit
submit = pd.DataFrame()
submit['PassengerId'] = test['PassengerId']
submit['Survived'] = prediction.astype(int)
submit.to_csv('submit.csv', index= False)


#first try ensemble = 0.772
#improve 1.feature engineer name->title 2.group age 3.repredict age
#second try after improvement using ensember  = 0.779
#Third try after feature rebuild = 0.791
