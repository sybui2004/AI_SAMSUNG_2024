 
import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score
from sklearn.metrics import explained_variance_score


def print_df(dataframe: pd.DataFrame):
    print(tabulate(dataframe, headers="keys", tablefmt='psql'))
    
df = pd.read_csv("Life Expectancy Data.csv")
print(df)
def check_null(dataframe: pd.DataFrame):
    return dataframe.isna().sum()

def rnd(dataframe: pd.DataFrame):
    dataframe.fillna(round(dataframe.mean()), inplace=True)


def numberic(dataframe: pd.DataFrame):
    df = dataframe.copy()
    for key in df.keys():
        if df[key].dtype == object:
            df[key] = LabelEncoder().fit_transform(df[key])
    return df

def stand(dataframe: pd.DataFrame):
    data = dataframe.copy()
    for key in data.keys():
        if data[key].dtype != object:
            data[key] = (data[key] - data[key].min()) / (data[key].max() - data[key].min()) 
    return data

for i in df.columns[:]:
    if check_null(df[i]) > 0:
        rnd(df[i])
        
data_life = df['Life expectancy ']
df.drop('Life expectancy ', inplace=True, axis=1)
df["Life expectancy"] = data_life 

df3 = df.copy()
df3 = numberic(df3)
print(df3)

df3.describe()

x_train, x_test, y_train, y_test  = train_test_split(df3.iloc[:, :-1], df3.iloc[:,-1], test_size=0.15, random_state=42)

x, y = df3.iloc[:, :-1], df3.iloc[:, -1]

lig = LinearRegression()
lig.fit(x, y)

y_predict = lig.predict(x)
print("Metric MSE: ", mean_squared_error(y_true=y, y_pred=y_predict, squared=True))
print("Metric MSA: ", mean_absolute_error(y_true=y, y_pred=y_predict))
print("Metric R2: ", r2_score(y_true=y, y_pred=y_predict) * 100, "\n")

lir = LinearRegression()

lir.fit(x_train, y_train)
y_pre = lir.predict(x_test)
y1_pre = lir.predict(x_train)
print("Train MSE: ", mean_squared_error(y_true=y_train, y_pred=y1_pre, squared=True))
print("Train MSA: ", mean_absolute_error(y_true=y_train, y_pred=y1_pre))
print("Train R2: ", r2_score(y_true=y_train, y_pred=y1_pre) * 100, "\n")

print("Test MSE: ", mean_squared_error(y_true=y_test, y_pred=y_pre, squared=True))
print("Test MSA: ", mean_absolute_error(y_true=y_test, y_pred=y_pre))
print("Test R2: ", r2_score(y_true=y_test, y_pred=y_pre) * 100)


plt.plot(y_pre, y_test, "ro")
plt.plot(y1_pre, y_train, "ro", c="b")
plt.show()


sc = StandardScaler()
x_train  = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
lirg = LinearRegression()
lirg.fit(x_train, y_train)

yA_pre = lirg.predict(x_test)
yA1_pre = lirg.predict(x_train)
print("Train MSE: ", mean_squared_error(y_true=y_train, y_pred=yA1_pre, squared=True))
print("Train MSA: ", mean_absolute_error(y_true=y_train, y_pred=yA1_pre))
print("Train R2: ", r2_score(y_true=y_train, y_pred=yA1_pre) * 100)
print("Train explained_variance_score: ",explained_variance_score(y_true=y_train, y_pred=yA1_pre), "\n")

print("Test MSE: ", mean_squared_error(y_true=y_test, y_pred=yA_pre, squared=True))
print("Test MSA: ", mean_absolute_error(y_true=y_test, y_pred=yA_pre))
print("Test R2: ", r2_score(y_true=y_test, y_pred=yA_pre) * 100)

plt.plot(yA_pre, y_test, 'ro')
plt.plot(yA1_pre, y_train, 'ro', color='b')
plt.plot()

logit = LinearRegression()
standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
scoring = ['r2', "neg_mean_squared_error", "neg_mean_absolute_error"]

pipeLine = make_pipeline(standard_scaler, logit)
kf = KFold(n_splits=5, shuffle=True, random_state=2)

cv_resutls = cross_validate(
    pipeLine, x_train, y_train, cv=kf, n_jobs=-1, scoring=scoring)

cv_resutls

print('test_r2: ', round(cv_resutls['test_r2'].mean(), 6))
print('test_neg_mean_absolute_error: ', round(cv_resutls['test_neg_mean_absolute_error'].mean(), 6))
print("test_neg_mean_squared_error: ", round(cv_resutls["test_neg_mean_squared_error"].mean(), 6))

x, y = df3.iloc[:, :2], df3.iloc[:, -1]
linear_regres = LinearRegression()
linear_regres.fit(x, y)

array_country = list(df['Country'].value_counts().index)
data_country = list(df['Country'])

def predict_to_year(year: int):
    data_year = pd.DataFrame({
    "Country": df3['Country'],
    "Year": pd.Series([year] * 2938),
    })
    return data_year

def mean_year(data_year_predict: list):
    dict_mean_country = dict()
    for i in array_country:
        dict_mean_country[i] = []
    for i in range(len(data_country)):
        dict_mean_country[data_country[i]].append(data_year_predict[i])
    array_country_old = []
    for i in array_country:
        array_country_old.append(float(round(pd.DataFrame(dict_mean_country[i]).mean(), 6)))
    return array_country_old

data_seaborn_country = list(df['Country'])
data_seaborn_year = list(df3['Year'])
data_seaborn_Life = list(df3['Life expectancy'])

for year in range(2016, 2031):
    data_year = predict_to_year(year)
    data_year_predict = linear_regres.predict(data_year)
    data_year_predict = list(data_year_predict)
    array_country_old = mean_year(data_year_predict)
    data_seaborn_country.extend(array_country)
    data_seaborn_year.extend([year] * 193)
    data_seaborn_Life.extend(array_country_old)
    
data_seaborn = pd.DataFrame({
    "Country" : pd.Series(data_seaborn_country),
    "Year" : pd.Series(data_seaborn_year),
    "Life expectancy" : pd.Series(data_seaborn_Life),
})

country_set = set(data_seaborn['Country'])

plt.figure()
plt.xlabel("year")
plt.ylabel("Life expectancy")
idx: int = 0
for country in country_set:
     selected_data = data_seaborn.loc[data_seaborn['Country'] == country]
     list_1 = list(selected_data['Year'])
     list_2 = list(selected_data['Life expectancy'])
     for i in range(len(list_1) - 1):
          for j in range(i + 1, len(list_1)):
               if list_1[i] < list_1[j]:
                    list_1[j], list_1[i] = list_1[i], list_1[j]
                    list_2[j], list_2[i] = list_2[i], list_2[j]
     plt.plot(list_1, list_2,label=country )
     if idx == 4:
          break
     idx += 1
     
plt.legend()
plt.show()


