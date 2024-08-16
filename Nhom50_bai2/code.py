import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

train_data_dir = 'BaiThi2_Samsung/TrainData'
test_data_dir = 'BaiThi2_Samsung/TestData_nolabel'

train_data = {'spam': [], 'notspam': []}

for label in train_data:
    label_dir = os.path.join(train_data_dir, label)
    for filename in os.listdir(label_dir):
        with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as file:
            train_data[label].append(file.read())

X_train = train_data['spam'] + train_data['notspam']
y_train = ['spam'] * len(train_data['spam']) + ['notspam'] * len(train_data['notspam'])

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)

clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

test_labels = []
test_files = os.listdir(test_data_dir)

for test_file in test_files:
    with open(os.path.join(test_data_dir, test_file), 'r', encoding='utf-8') as file:
        test_content = file.read()
        test_content_counts = vectorizer.transform([test_content])
        predicted_label = clf.predict(test_content_counts)[0]
        test_labels.append((test_file, predicted_label))

result_df = pd.DataFrame(test_labels, columns=['Filename', 'Label'])

result_df.to_csv('TestLabels.csv', index=False)

print("Done")
