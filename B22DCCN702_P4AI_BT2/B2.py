import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df = pd.read_csv('AB_NYC_2019 (1).csv')
#1 Vẽ biểu đồ cột cho neighborhood group
neighbourhood_group_availability = df.groupby('neighbourhood_group')['availability_365'].mean()
plt.figure(figsize=(10, 6))
neighbourhood_group_availability.plot(kind='bar', color='skyblue')
plt.title('Trung bình availability_365 theo neighbourhood group')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Trung bình Availability 365')
plt.xticks(rotation=45)
plt.show()

#2 Tạo histogram cho neighbourhood
neighbourhood_counts = df['neighbourhood'].value_counts()
plt.subplot(1, 2, 1)
plt.hist(neighbourhood_counts, bins=100, color='skyblue')
plt.title("Histogram for Neighbourhood (Counts)")
plt.xlabel('Neighbourhood')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


#3 Biểu diễn mối liên hệ giữa neighbourgroup và Availability của các phòng

neighbourhood_group_availability = df.groupby('neighbourhood_group')['availability_365'].sum()

sns.barplot(x=neighbourhood_group_availability.index, y=neighbourhood_group_availability.values, palette='viridis')
plt.title('Tổng số ngày khả dụng (availability_365) theo neighbourhood group')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Tổng Availability 365')
plt.xticks(rotation=45)
plt.show()

#4 Vẽ bản đồ (scatter plot) của neighborhood dựa theo tọa độ lat lon
plt.scatter(df['longitude'], df['latitude'], c='blue', alpha=0.5)
plt.title('Scatter Plot của Neighbourhood dựa theo tọa độ Latitude và Longitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

#5 Sử dụng heatmap để biểu diễn mối quan hệ (correlation) giữa tất cả các thuộc tính trong dữ liệu

numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap biểu diễn mối quan hệ giữa các thuộc tính')
plt.show()