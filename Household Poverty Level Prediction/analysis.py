import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')

train.boxplot(by='Target', column=['meaneduc'], grid=True)
# plt.show()

categoricals = [column for column in train.columns if train[column].dtype == 'object']
print(categoricals)

print(train[[categoricals[1], categoricals[2], categoricals[3], categoricals[4]]].head())
