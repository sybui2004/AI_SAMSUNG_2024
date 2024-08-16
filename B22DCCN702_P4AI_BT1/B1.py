import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

def show_DataFrame(df):
    root = tk.Tk()
    root.title("P4AI_BT1")

    frame = ttk.Frame(root)
    frame.pack(fill='both', expand=True)

    tree = ttk.Treeview(frame, columns=list(df.columns), show='headings')
    tree.pack(side='left', fill='both', expand=True)

    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    for _, row in df.iterrows():
        tree.insert('', 'end', values=list(row))

    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side='right', fill='y')

    root.mainloop()
    
# 1
data = pd.read_csv('P4AI_BT1.csv')

show_DataFrame(data)
# 2
data_frame = data.copy()

data_frame['sepal.length'].fillna(round(data['sepal.length'].mean()), inplace=True)
data_frame['sepal.width'].fillna(round(data['sepal.width'].mean()), inplace=True)
data_frame['petal.length'].fillna(round(data['petal.length'].mean()), inplace=True)
data_frame['petal.width'].fillna(round(data['petal.width'].mean()), inplace=True)

data_frame['variety'].fillna(data_frame['variety'].mode()[0], inplace = True)

show_DataFrame(data_frame)

#3

data_frame_1 = data_frame[(data_frame['sepal.length'] > 5) & (data_frame['sepal.width'] > 3)]
show_DataFrame(data_frame_1)

#4

scaler = MinMaxScaler()
data_frame_1[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']] = scaler.fit_transform(data_frame_1[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']])

show_DataFrame(data_frame_1)

#5
data_frame_2 = data_frame.copy()
data_frame_2['variety'].unique()
dummies = pd.get_dummies(data_frame_2['variety'])
data_frame_2 = pd.concat([data_frame_2,dummies], axis=1)
data_frame_2.drop('variety', axis=1, inplace=True)

show_DataFrame(data_frame_2)

#6
# X = data_frame_2.drop('variety', axis=1)
# y = data_frame_2['variety']

# model = LinearRegression()
# model.fit(X, y)

# model.predict(X)
#7

sample_data = data_frame.sample(frac=0.5, replace=True, random_state=1)

show_DataFrame(sample_data)

#8

data_frame['sepal.length.sporadic'] = pd.cut(data_frame['sepal.length'], bins=3, labels=False)

show_DataFrame(data_frame)

#9

kmeans = KMeans(n_clusters = 3, random_state = 1)
data_frame['cluster'] = kmeans.fit_predict(data_frame.drop('variety', axis = 1))
iso_forest = IsolationForest(contamination = 0.1, random_state = 1)
data_frame['outlier'] = iso_forest.fit_predict(data_frame.drop(['variety', 'cluster'], axis = 1))
data_without_outliers = data_frame[data_frame['outlier'] == 1]
show_DataFrame(data_without_outliers)
