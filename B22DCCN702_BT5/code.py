import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN, KMeans
from dpca import DensityPeakCluster
from sklearn import metrics

data = pd.read_csv("countries.csv")
print(data)

plt.scatter(data['latitude'], data['longitude'], c='red')
plt.show()

import os, googlemaps
from pymaps.pymaps import Map
lat, lon = 8.619543, 0.824782
data = pd.read_csv('countries.csv')
print(data)

import gmplot

latitude_list = data['latitude']
longitude_list = data['longitude']

gmap = gmplot.GoogleMapPlotter(0, 0, 2) 

gmap.scatter(latitude_list, longitude_list, '#FF0000', size=40, marker=False)

gmap.draw("map.html")


data.isna().sum()
def rnd(df):
    # round 
    df.fillna(round(df.mean()), inplace=True)
rnd(data['latitude'])
rnd(data['longitude'])

kmeans = KMeans(n_clusters = 5)
y_pred = kmeans.fit_predict(data.iloc[:,[1,2]])

plt.figure(figsize=(20,12))
plt.scatter(
    data['latitude'], data['longitude'],
    c=y_pred, marker='o',
    edgecolor='black', s=200
)
plt.xlabel("X", fontsize=20)
plt.ylabel("Y", fontsize=20)
plt.title(" KMeans", fontsize=40)
plt.show()

db = DBSCAN(eps=10, min_samples=5).fit(data.iloc[:, [1, 2]])
y_pred_DB = db.fit_predict(data.iloc[:, [1, 2]])
plt.figure(figsize=(20,12))
plt.scatter(
    data['latitude'], data['longitude'],
    c=y_pred_DB, marker='o',
    edgecolor='black', s=200
)
plt.xlabel("X",fontsize=20)
plt.ylabel("Y",fontsize=20)
plt.title(" DBSCAN", fontsize=40)
plt.show()

dpca = DensityPeakCluster(density_threshold=10, distance_threshold= 7, anormal=False)
dpca.fit(data.iloc[:,[1,2]])
rho = dpca.local_density()
delta, nneigh = dpca.min_neighbor_and_distance()
labels, center = dpca.collapse()

plt.figure(figsize=(20,12))
plt.scatter(
    rho,delta,
    c=labels, marker='o',
    edgecolor='black', s=200
)
plt.xlabel("Density", fontsize=20)
plt.ylabel("Distance", fontsize=20)
plt.title(" Decision graph", fontsize=40)
plt.show()

plt.figure(figsize=(20,12))
plt.scatter(
    data['latitude'],data['longitude'],
    c=labels, marker='o',
    edgecolor='black', s=200
)
plt.xlabel("X", fontsize=20)
plt.ylabel("Y", fontsize=20)
plt.title(" DPC", fontsize=40)
plt.show()

# So sánh kết quả rồi nhận xét:
# trong 3 thuật toán Kmeans ổn nhất nhưng khó bỏ ngoại lai
# DBSCAN khó phân loại
# DPC chia khá ổn nhưng khó