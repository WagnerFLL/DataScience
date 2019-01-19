import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier

Role = {
    "Capt": "crew", "Col": "crew", "Major": "crew",
    "Jonkheer": "royalty", "Don": "royalty", "Sir": "royalty",
    "Dr": "crew", "Rev": "crew", "the Countess": "royalty",
    "Mme": "married", "Mlle": "single", "Ms": "married",
    "Mr": "Mr", "Mrs": "married", "Miss": "single",
    "Master": "master", "Lady": "royalty", "Dona": "married"
}


def get_role(data):
    role = []
    for index in data.index:
        role.append(data['Name'][index].split(',')[1].split('.')[0].strip())

    data['Role'] = list(map(lambda name: Role[name], role))

    return data


def infer_age(data):
    new_data = data.dropna()
    age_dictionary = {
        'master': new_data.loc[new_data['Role'] == 'master'].median().Age,
        'crew': new_data.loc[new_data['Role'] == 'crew'].median().Age,
        'royalty': new_data.loc[new_data['Role'] == 'royalty'].median().Age,
        'single': new_data.loc[new_data['Role'] == 'single'].median().Age,
        'married': new_data.loc[new_data['Role'] == 'married'].median().Age,
        'Mr': new_data.loc[new_data['Role'] == 'Mr'].median().Age
    }

    for index in data.index:
        if math.isnan(data['Age'][index]):
            data['Age'][index] = age_dictionary[data['Role'][index]]

    return data


def process_cabin(data):
    data.Cabin.fillna('D', inplace=True)
    data['Cabin'] = data['Cabin'].map(lambda cabin: cabin[0])

    return data


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train = get_role(train)
train = infer_age(train)
train = process_cabin(train)
train.dropna(axis=0, how='any', inplace=True)
train.drop(['Name', 'Ticket'], axis=1, inplace=True)
train = pd.get_dummies(train)

test = get_role(test)
test = infer_age(test)
test = process_cabin(test)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)
test.drop(['Name', 'Ticket'], axis=1, inplace=True)
test = pd.get_dummies(test)

X_train = train.drop('Survived', axis=1)
Y_train = train['Survived']

forest = RandomForestClassifier(n_estimators=180, min_samples_leaf=3, max_features=0.5, n_jobs=-1)
forest.fit(X_train, Y_train)

test['Cabin_T'] = 0
test['Role_royalty'] = 0

sub = pd.DataFrame()
sub['PassengerId'] = test['PassengerId']
sub['Survived'] = forest.predict(test)
sub.to_csv('submission.csv', index=False)
