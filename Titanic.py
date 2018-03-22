import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals.six import StringIO

train = pd.read_csv('train.csv')
kaggle = pd.read_csv('test.csv')

PassengerId_train = train['PassengerId'].tolist()
PassengerId = kaggle['PassengerId'].tolist()

def group_female_titles(elem):
    if elem in ['Ms','Miss','Mme','Mlle','Lady','Dona','Countess']:
        return 'Mrs'
    return elem

def set_title(df):
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].map(group_female_titles)
    return df


def set_age_group(df):
    bins = [0, 3, 12, 18, 60, 100]
    labels = ['Baby', 'Child', 'Teenager', 'Adult', 'Elder']
    age_groups = pd.cut(df.Age, bins, labels=labels)
    df['Age_group'] = age_groups
    return pd.concat([df, pd.get_dummies(df['Age_group'])], axis=1)

def set_is_alone(df):
    for i, row in df.iterrows():
        if ((row['Parch']==0) and (row['SibSp']==0)):
            val = 1
        else:
            val = 0
        df.set_value(i,'is_alone',val)
    return df

def set_family_size(df):
    for i, row in df.iterrows():
        df.set_value(i,'family_size',row['Parch']+row['SibSp'])
    return df

def process_data(df):
    df = df.drop('Cabin', axis=1)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    #df = set_age_group(df)
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(), inplace=True)
    portName = {'Q':'Queenstown', 'C':'Cherbourg', 'S':'Southampton'}
    df['Embarked'] = df['Embarked'].map(portName)
    df = pd.concat([df, pd.get_dummies(df['Embarked'])], axis=1)
    df['Sex'] = df['Sex'].map({'female':0, 'male':1})
    df = set_is_alone(df)
    df = set_family_size(df)
    return df

#prépare le tableau de train et le tableau de test
train = process_data(train)
kaggle = process_data(kaggle)

'''
train = set_title(train)
train['Sexe'] = train['Sex'].map({0:'Femme',1:'Homme'})
#kaggle.sort_values(['Sex','Pclass','Age'])
plt.figure()
sns.barplot(x='Embarked',y='Fare',hue='Pclass',data=train)
plt.show()
plt.figure()
sns.countplot(x='Embarked',hue='Pclass',data=train)
plt.show()
plt.figure()
sns.countplot(x='Embarked',hue='family_size',data=train)
plt.show()
'''

#prépare un tableau avec le train + le test
train_kaggle = train.append(kaggle)
train_kaggle = set_title(train_kaggle)
#créé les colonnes "onehot" pour les titres
title_onehot = pd.get_dummies(train_kaggle['Title'])
liste_titles_onehot = title_onehot.keys()
title_onehot['PassengerId'] = train_kaggle['PassengerId']
#ajoute les colonnes "onehot" pour les titres aux tableaux de train et de test
train = train.merge(title_onehot.iloc[:len(train),:], how='outer', on='PassengerId')
kaggle = kaggle.merge(title_onehot.iloc[-len(kaggle):,:], how='outer', on='PassengerId')

#calcul et affichage des facteurs de corrélation des variables
correlation = train.corr()
correlation = correlation['Survived'].abs().sort_values(ascending= False)
print('correlation = '+"\n",correlation)

#suppression des colonnes non pertinentes
cols = ['SibSp', 'PassengerId','Embarked']
train = train.drop(cols, axis = 1)
kaggle = kaggle.drop(cols, axis = 1)

#sélection des colonnes pertinentes
#X_cols = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'is_alone', 'Southampton', 'Cherbourg', 'Queenstown']
X_cols = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'is_alone', 'family_size', 'Southampton', 'Cherbourg', 'Queenstown']
X_cols.extend(liste_titles_onehot)

#création des tableaux d'entrainement et de cible, des tableaux de validation intermédiaire et le tableau à prédire au final
X = train[X_cols]
y = train['Survived']
X_kaggle = kaggle[X_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#étalonnage des valeurs
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_kaggle = scaler.transform(X_kaggle)

names = [
        'Logistic Regression',
        'Decision Tree',
        #'Gaussian Naive Bayes',
        'MPLCLassifier',
        #'QuadraticDiscriminantAnalysis',
        'KNeighborsClassifier',
        'SVC linear 0.025',
        'SVC gamma=2 C=1',
        'SVC gamma=3 C=3',
        'GaussianProcessClassifier',
        'RandomForestClassifier',
        'AdaBoostClassifier']
classifier = [
        LogisticRegression(C=3),
        tree.DecisionTreeClassifier(),
        #GaussianNB(),
        MLPClassifier(alpha=1),
        #QuadraticDiscriminantAnalysis(),
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        SVC(gamma=3, C=3),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier()]

#fonction de cross_validation
best_clf = None
best_clf_name = None
best_score = 0
idx = 0
clf_results = pd.DataFrame([{'name':'','score':0} for x in range(10)])
clf_results = pd.DataFrame(index=range(10), columns=['name','score'])
def run_kfold(clf, name, best_clf, best_score, best_clf_name, clf_results):
    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print(name, " - Fold {0} accuracy: {1}".format(fold, accuracy))
    mean_outcome = np.mean(outcomes)
    std_outcome = np.std(outcomes)
    print(name," - Mean Accuracy: {0}".format(mean_outcome),' +/- ',std_outcome)
    if mean_outcome > best_score:
        best_score = mean_outcome
        best_clf = clf
        best_clf_name = name
    #clf_results[idx]['name'] = name
    #clf_results[idx]['score'] = mean_outcome
    clf_results.set_value(idx,'name',name)
    clf_results.set_value(idx,'score',mean_outcome)
    return (best_clf, best_score, best_clf_name)

print(clf_results)
#entrainement des différents classifieurs
for name, clf in zip(names,classifier):
    best_clf, best_score, best_clf_name = run_kfold(clf, name, best_clf, best_score, best_clf_name, clf_results)
    idx+=1

plt.figure()
sns.barplot(x='score',y='name',data=clf_results)
plt.show()
print('Best classifier = ',best_clf_name)
print('with score = ',best_score)

#préparation des différents paramètres du classifieur à tester
parameters['Logistic Regression'] = {'C': [1, 3, 6, 10],
              'dual': [False, True],
              'penalty': ['l2', 'l1'],
              'fit_intercept': [False, True],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
             }
'''parameters['Logistic Regression'] = {'C': [1, 3, 6, 10],
              'dual': [False],
              'penalty': ['l2'],
              'fit_intercept': [False, True],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
             }'''
parameters['GaussianProcessClassifier'] = {'n_restarts_optimizer': [0, 1, 2],
              'warm_start': [False, True],
              'random_state': [None, 2]
             }
parameters['RandomForestClassifier'] = {'n_estimators': [4, 6, 9],
              'max_features': ['log2', 'sqrt','auto'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }
parameters['AdaBoostClassifier'] = {'n_estimators': [50, 25, 125],
              'learning_rate': [0.5,1,2],
              'algorithm': ['SAMME', 'SAMME.R'],
              'random_state': [None, 2]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)
# Run the grid search
grid_obj = GridSearchCV(best_clf, parameters[best_clf_name], scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)
# Set the clf to the best combination of parameters
best_clf = grid_obj.best_estimator_
# Fit the best algorithm to the data.
best_clf.fit(X_train, y_train)
#création des prédictions pour la validation intermédiaire
predictions = best_clf.predict(X_test)
print('Accuracy on test set = ', accuracy_score(y_test, predictions))


#création des prédictions sur l'échantillon à tester
predictions = best_clf.predict(X_kaggle)

#création du tableau de sortie pour Kaggle
output = pd.DataFrame({ 'PassengerId' : PassengerId, 'Survived': predictions })
output = output.set_index('PassengerId')
#output.to_csv('data.csv')

#hack pour l'affichage
kaggle['PassengerId'] = PassengerId
kaggle['Survived'] = predictions

'''
#pour afficher les stats selon les données d'entraînement
data = pd.DataFrame( {'PassengerId' : PassengerId_train, 'Survived' : train['Survived']} )
data = data.set_index('PassengerId')
kaggle = train
kaggle['PassengerId'] = PassengerId_train
kaggle['Survived'] = train['Survived']
'''

liste_titres = train_kaggle['Title'].value_counts().keys()
dict_titres = dict(zip(liste_titres,range(len(liste_titres))))
kaggle = set_title(kaggle)
kaggle['Title_id'] = kaggle['Title'].map(dict_titres)

kaggle['Sexe'] = kaggle['Sex'].map({0:'Femme',1:'Homme'})
#kaggle.sort_values(['Sex','Pclass','Age'])

plt.figure()
sns.barplot(x='Pclass',y='Survived',hue='Sexe',data=kaggle)
plt.figure()

sns.barplot(x='Title',y='Survived',hue='Sexe',data=kaggle)
plt.figure()

sns.barplot(x='Parch',y='Survived',hue='Sexe',data=kaggle)
plt.figure()

sns.barplot(x='is_alone',y='Survived',hue='Sexe',data=kaggle)
plt.figure()

sns.barplot(x='family_size',y='Survived',hue='Sexe',data=kaggle)
plt.figure()

#affichage du modèle appris pour la survie selon le prix du billet et le sexe
bins = [0, 15, 30, 60, 80, 500]
labels = ['0-15', '15-30', '30-60', '60-80', '+80']
fare_groups = pd.cut(kaggle.Fare, bins, labels=labels)
kaggle['fare_group_labeled'] = fare_groups
sns.barplot(x='fare_group_labeled',y='Survived',hue='Sexe',data=kaggle)
plt.figure()

#affichage du modèle appris pour la survie selon le port d'embarcation et le sexe
def get_embark(item):
    if item['Southampton'] == 1:
        return 'Southampton'
    elif item['Queenstown'] == 1:
        return 'Queenstown'
    elif item['Cherbourg'] == 1:
        return 'Cherbourg'
kaggle['Embarked'] = kaggle.apply(get_embark, axis=1)
sns.barplot(x='Embarked',y='Survived',hue='Sexe',data=kaggle)
plt.figure()

#affichage du modèle appris pour la survie selon l'âge et le sexe
bins = [0, 3, 12, 18, 55, 100]
labels = ['baby', 'child', 'teenager', 'adult', 'elder']
age_groups = pd.cut(kaggle.Age, bins, labels=labels)
kaggle['Age_group'] = age_groups
sns.barplot(x='Age_group',y='Survived',hue='Sexe',data=kaggle)
plt.show()

#sns.violinplot(x = "Fare", data=kaggle)
#plt.show()
'''
sns.violinplot(x = "Age", data=kaggle)
plt.figure()
bins = [0, 3, 6, 9, 12, 15, 18, 55, 100]
bins = np.arange(0, 81, 3)
print(bins)
#labels = ['baby', 'child', 'teenager', 'adult', 'elder']
age_groups = pd.cut(kaggle.Age, bins)
kaggle['Age_group2'] = age_groups
sns.barplot(x="Age_group2", y="Survived", hue='Sexe', data=kaggle);
plt.show()'''
