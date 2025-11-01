import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

fake_data = pd.read_csv('Fake.csv')
true_data = pd.read_csv('True.csv')

# print(fake_data.head())
# print(true_data.head())

fake_data["class"] = 0
true_data["class"] = 1

# print(fake_data.shape, true_data.shape )

fake_data_manual_testing = fake_data.tail()
for i in range(23480, 23470, -1):
    fake_data.drop([i], axis = 0, inplace = True)
    
true_data_manual_testing = true_data.tail()
for i in range(21416, 21406, -1):
    true_data.drop([i], axis = 0, inplace = True)
    
# print(fake_data_manual_testing.shape, true_data_manual_testing.shape)

fake_data_manual_testing["class"] = 0
true_data_manual_testing["class"] = 1

# print(fake_data_manual_testing.head(10))
# print(true_data_manual_testing.head(10))

data_merge = pd.concat([fake_data, true_data], axis=0)
# print(data_merge.head(10))

# print(data_merge.columns)

data = data_merge.drop(['title','subject','date'], axis=1)
# print(data.isnull().sum())

data = data.sample(frac = 1)
# print(data.head())

data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)

# print(data.columns)
# print(data.head())

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+',b'',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    return text

data['text'] = data['text'].apply(wordopt)

x = data['text']
y = data['class']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)

print(LR.score(xv_test, y_test))
print (classification_report(y_test, pred_lr))


from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)

print(DT.score(xv_test, y_test))
print (classification_report(y_test, pred_lr))

from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train, y_train)

pred_gb = GB.predict(xv_test)

print(GB.score(xv_test, y_test))
print(classification_report(y_test, pred_gb))

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state = 0)

pred_rf = RF.predict(xv_test)
RF.fit(xv_train, y_train)

RF.score(xv_test, y_test)
print (classification_report(y_test, pred_rf))