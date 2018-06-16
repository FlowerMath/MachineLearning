import pandas

import numpy as np

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
print(data.head(1))
count_female = data['Sex'].value_counts()['female']
count_male = data['Sex'].value_counts()['male']
print('female = ', count_female, ', ', 'male = ', count_male)

all = data.shape[0]
surv = data['Survived'].value_counts()[1]
print(data['Survived'].value_counts())
print(round(100 * surv / all,2), '%')

first_class = data[data.Pclass == 1].shape[0]
print(round(100 * first_class / all, 2), '%')

avr_age = data['Age'].mean()
med_age = data['Age'].median()
print(round(avr_age, 4), ', ', med_age)

datacorr = data[['SibSp', 'Parch']]
correlation = datacorr.corr(method='pearson')
print(round(correlation, 2))

female_names = data[data.Sex == 'female']['Name']
print(female_names.head(12))

female_names1 = data[data.Sex == 'female']['Name'].str.split('.', expand=True)[1]

married_female_names = female_names1[female_names1.str.contains('\(')].str.split('\(', expand=True)[1].str.replace('\)',
                                                                                                                   '')

married_female_names_one = married_female_names.str.split(' ', expand=True)[0]
print(married_female_names_one.head(6))
single_female_names = female_names1[~female_names1.str.contains('\(')]
single_female_names_one = single_female_names.str.split(' ', expand=True)[1]
print(single_female_names_one.head(6))

female_first_names = married_female_names_one.append(single_female_names_one)
female_first_names.to_frame()
print(female_first_names.value_counts().head(5))


