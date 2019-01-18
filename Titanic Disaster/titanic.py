import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

data_train = pd.get_dummies(train)
data_test = pd.get_dummies(test)

data_train['Age'].fillna(data_train['Age'].mean(), inplace=True)
data_test['Age'].fillna(data_test['Age'].mean(), inplace=True)
data_test['Fare'].fillna(data_test['Fare'].mean(), inplace=True)

# print(data_test.isnull().sum())

X_train = data_train.drop('Survived', axis=1)
Y_train = data_train['Survived']

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, Y_train)

print(tree.score(X_train, Y_train))

sub = pd.DataFrame()
sub['PassengerId'] = data_test['PassengerId']
sub['Survived'] = tree.predict(data_test)
sub.to_csv('submission.csv', index=False)