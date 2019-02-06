import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


def replace_yes_no(df):
    df[['dependency', 'edjefe', 'edjefa']] = df[['dependency', 'edjefe', 'edjefa']].replace({'yes': 1, 'no': 1}).astype(
        float)
    return df


def tablets(df):
    df['v18q1'][df['v18q1'].isnull()] = 0
    df = df.drop(['v18q'], axis=1)
    return df


def escolari(df):
    escolari_mean = df.groupby(['idhogar'], as_index=False)['escolari'].mean().rename(columns={'mean': 'escolari_mean'})
    escolari_mean.columns = ['idhogar', 'escolari_mean']

    escolari_max = df.groupby(['idhogar'], as_index=False)['escolari'].max().rename(columns={'max': 'escolari_max'})
    escolari_max.columns = ['idhogar', 'escolari_max']

    df = df.merge(escolari_mean, how='left', on='idhogar')
    df = df.merge(escolari_max, how='left', on='idhogar')

    return df


def water_provision(df):
    df['water_prov'] = 0
    df.loc[df['abastaguadentro'] == 1, 'water_prov'] = 2
    df.loc[df['abastaguafuera'] == 1, 'water_prov'] = 1
    df.loc[df['abastaguano'] == 1, 'water_prov'] = 0
    df = df.drop(['abastaguadentro', 'abastaguafuera', 'abastaguano'], axis=1)
    return df


def walls_roof_floor(df):
    df['walls'] = 0
    df.loc[df['epared1'] == 1, 'walls'] = 1
    df.loc[df['epared2'] == 1, 'walls'] = 2
    df.loc[df['epared3'] == 1, 'walls'] = 3

    df['roof'] = 0
    df.loc[df['etecho1'] == 1, 'roof'] = 1
    df.loc[df['etecho2'] == 1, 'roof'] = 2
    df.loc[df['etecho3'] == 1, 'roof'] = 3

    df['floor'] = 0
    df.loc[df['eviv1'] == 1, 'floor'] = 1
    df.loc[df['eviv2'] == 1, 'floor'] = 2
    df.loc[df['eviv3'] == 1, 'floor'] = 3

    df = df.drop(['epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3'], axis=1)

    return df


def education_level(df):
    df['education'] = 0
    df.loc[df['instlevel1'] == 1, 'education'] = 1
    df.loc[df['instlevel2'] == 1, 'education'] = 2
    df.loc[df['instlevel3'] == 1, 'education'] = 3
    df.loc[df['instlevel4'] == 1, 'education'] = 4
    df.loc[df['instlevel5'] == 1, 'education'] = 5
    df.loc[df['instlevel6'] == 1, 'education'] = 6
    df.loc[df['instlevel7'] == 1, 'education'] = 7
    df.loc[df['instlevel8'] == 1, 'education'] = 8
    df.loc[df['instlevel9'] == 1, 'education'] = 9

    df = df.drop(
        ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8',
         'instlevel9'], axis=1)

    return df


def tipovivi(df):
    df['tipovivi'] = 0
    df.loc[df['tipovivi1'] == 1, 'tipovivi'] = 1
    df.loc[df['tipovivi2'] == 1, 'tipovivi'] = 2
    df.loc[df['tipovivi3'] == 1, 'tipovivi'] = 3
    df.loc[df['tipovivi4'] == 1, 'tipovivi'] = 4
    df.loc[df['tipovivi5'] == 1, 'tipovivi'] = 5

    df = df.drop(['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5'], axis=1)

    return df


def rubbish(df):
    df['rubbish'] = 0
    df.loc[df['elimbasu1'] == 1, 'rubbish'] = 1
    df.loc[df['elimbasu2'] == 1, 'rubbish'] = 2
    df.loc[df['elimbasu3'] == 1, 'rubbish'] = 3
    df.loc[df['elimbasu4'] == 1, 'rubbish'] = 4
    df.loc[df['elimbasu5'] == 1, 'rubbish'] = 5
    df.loc[df['elimbasu6'] == 1, 'rubbish'] = 0

    df = df.drop(['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6'], axis=1)

    return df


def energy(df):
    df['energy'] = 0
    df.loc[df['energcocinar1'] == 1, 'energy'] = 1
    df.loc[df['energcocinar2'] == 1, 'energy'] = 2
    df.loc[df['energcocinar3'] == 1, 'energy'] = 3
    df.loc[df['energcocinar4'] == 1, 'energy'] = 4

    df = df.drop(['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4'], axis=1)

    return df


def toilet(df):
    df['toilet'] = 0
    df.loc[df['sanitario1'] == 1, 'toilet'] = 1
    df.loc[df['sanitario5'] == 1, 'toilet'] = 2
    df.loc[df['sanitario6'] == 1, 'toilet'] = 3
    df.loc[df['sanitario3'] == 1, 'toilet'] = 4
    df.loc[df['sanitario2'] == 1, 'toilet'] = 5

    df = df.drop(['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6'], axis=1)

    return df


def new_variables(df):
    df['rent_by_hhsize'] = df['v2a1'] / df['hhsize']  # rent by household size
    df['rent_by_people'] = df['v2a1'] / df['r4t3']  # rent by people in household
    df['rent_by_rooms'] = df['v2a1'] / df['rooms']  # rent by number of rooms
    df['rent_by_living'] = df['v2a1'] / df['tamviv']  # rent by number of persons living in the household
    df['rent_by_minor'] = df['v2a1'] / df['hogar_nin']
    df['rent_by_adult'] = df['v2a1'] / df['hogar_adul']
    df['children_by_adults'] = df['hogar_nin'] / df['hogar_adul']
    df['house_quali'] = df['walls'] + df['roof'] + df['floor']
    df['tablets_by_adults'] = df['v18q1'] / df['hogar_adul']  # number of tablets per adults
    df['ratio_nin'] = df['hogar_nin'] / df['hogar_adul']  # ratio children to adults
    return df


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

categorical = [column for column in train.columns if train[column].dtype == 'object']
print(categorical)

xg = XGBClassifier(booster="gbtree", learning_rate=0.2, min_split_loss=0,
                   reg_lambda=1, reg_alpha=0, tree_method="exact")

X = train.drop('Target', axis=1, inplace=False)
X = replace_yes_no(X)
X = new_variables(toilet(energy(rubbish(tipovivi(education_level(walls_roof_floor(water_provision(escolari(tablets(X))))))))))
X.drop(['Id','idhogar'], axis=1, inplace=True)
Y = train.Target
xg.fit(X, Y)

X = replace_yes_no(test)
X = new_variables(toilet(energy(rubbish(tipovivi(education_level(walls_roof_floor(water_provision(escolari(tablets(X))))))))))
X.drop(['Id','idhogar'], axis=1, inplace=True)
preds = xg.predict(X)

subs = pd.DataFrame()
subs['Id'] = test['Id']
subs['Target'] = preds
subs.to_csv('submission.csv', index=False)
