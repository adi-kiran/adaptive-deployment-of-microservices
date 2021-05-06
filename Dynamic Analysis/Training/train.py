import pandas as pd
import os
import csv
from glob import glob
import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle

pathdir = os.listdir('drive/MyDrive/Dataset_ping')
print(pathdir)
print(len(pathdir))

PATH = "drive/MyDrive/Dataset_ping"
EXT = "*.csv"
all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]

print(all_csv_files[:10])
print(len(all_csv_files))

df = pd.DataFrame()
for name in all_csv_files:
    ground_truth = 0
    if '/Trai' in name:
        ground_truth = 1
    elif '/Atta' in name:
        ground_truth = 0
    df1 = pd.read_csv(name)
    new_dict = dict()
    try:
        for index, row in df1.iterrows():
            new_dict[row['syscall']] = row['count']
    except:
        print(name)
        raise KeyError


    new_dict['Truth'] = ground_truth
    df =  df.append(new_dict, ignore_index=True)

df = df.fillna(0)
df = df.sample(frac=1)

X = df.loc[:, df.columns != 'Truth']
y = df['Truth']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print("Precision:",metrics.precision_score(Y_test, y_pred))
print("Recall:",metrics.recall_score(Y_test, y_pred))

# save the model to disk
filename = 'finalized_model.sav'


