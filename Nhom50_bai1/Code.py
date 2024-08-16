import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

train_data = pd.read_csv('Train_samsung.csv')
test_data = pd.read_csv('Test_samsung_noclass.csv')

predict_data = test_data.copy()

list_data = [train_data, test_data]

replace_dicts = {
    'X3': {"3+": 3, "0": 0, "1": 1, "2": 2},
    'X1': {"Male": 1, "Female": 0},
    'X2': {"Yes": 1, "No": 0},
    'X5': {"Yes": 1, "No": 0},
    'X4': {"Graduate": 1, "Not Graduate": 0},
    'X11': {"Semiurban": 0, "Rural": 1, "Urban": 2},
    'Class': {"Y": 1, "N": 0}
}

for data in list_data:
    for column, replace_dict in replace_dicts.items():
        if column in data.columns:
            data[column].replace(replace_dict, inplace=True)

def clean_fre(df):
    cnt = df.value_counts(dropna=True)
    df.fillna(cnt.index[0], inplace=True)
    
def rnd(df):
    df.fillna(round(df.mean()), inplace=True)

for data in list_data:
    for i in range(len(data.columns) - 1):
        rnd(data[data.columns[i]])

for data in list_data:
    for i in range(5, 9):
        data[data.columns[i]] = (data[data.columns[i]] - data[data.columns[i]].min()) / (data[data.columns[i]].max() - data[data.columns[i]].min())

X_train, X_test, y_train, y_test = train_test_split(train_data.iloc[:, :-1], train_data.iloc[:, -1], test_size=0.12, random_state=42)

clf = RidgeClassifier(alpha=1.0, random_state=6933)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred, average='weighted'))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))

x_test_samsung = test_data.copy()
y_test_samsung = clf.predict(x_test_samsung)
predict_data["Class"] = y_test_samsung
predict_data['Class'].replace({1: "Y", 0: "N"}, inplace=True)

predict_data.to_csv("class_label.csv", index=False, columns=["Class"])
print(predict_data)
