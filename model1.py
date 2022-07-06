import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import AdaBoostClassifier


datasets = pd.read_csv('immune.csv')

X= datasets.iloc[:,:7]

y = datasets.iloc[:, -1]


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
sc=StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.fit(X_test)

boost = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
model1 = boost.fit(X_train, y_train)
y_pred = model1.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

pickle.dump(model1,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

