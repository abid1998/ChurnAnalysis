import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score , GridSearchCV
from keras.layers import Dropout


dataset = pd.read_csv('Churn_Modelling.csv')
dataset.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

LabelEncoder_X = LabelEncoder()
X[:,1] = LabelEncoder_X.fit_transform(X[:,1])

LabelEncoder_X1 = LabelEncoder()
X[:,2] = LabelEncoder_X1.fit_transform(X[:,2])

OneHotEncoder_X = OneHotEncoder(categorical_features=[1])
X = OneHotEncoder_X.fit_transform(X).toarray()
X = X[:,1:]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

stsc = StandardScaler()

X_train = stsc.fit_transform(X_train)
X_test = stsc.fit_transform(X_test)
'''
clf = Sequential()

clf.add(Dense(output_dim = 6,init='uniform', activation = 'relu',input_dim=11))
clf.add(Dense(output_dim = 6,init='uniform', activation = 'relu'))
clf.add(Dense(output_dim = 1,init='uniform', activation = 'sigmoid'))

clf.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics= ['accuracy'])

clf.fit(X_train,y_train,batch_size=10,nb_epoch=100 )

y_pred = clf.predict(X_test)
y_pred = (y_pred>0.5)

print(confusion_matrix(y_test,y_pred))
'''
'''
clf = Sequential()

clf.add(Dense(output_dim = 6,init='uniform', activation = 'relu',input_dim=11))
clf.add(Dropout(p=0.1))

clf.add(Dense(output_dim = 6,init='uniform', activation = 'relu'))
clf.add(Dropout(p=0.1))

clf.add(Dense(output_dim = 1,init='uniform', activation = 'sigmoid'))
clf.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics= ['accuracy'])

clf.fit(X_train,y_train,batch_size=10,nb_epoch=100 )
'''
'''
 def build_classifier():
    clf = Sequential()

    clf.add(Dense(output_dim = 6,kernel_initializer='uniform', activation = 'relu',input_dim=11))
    clf.add(Dense(output_dim = 6,kernel_initializer='uniform', activation = 'relu'))
    clf.add(Dense(output_dim = 1,kernel_initializer='uniform', activation = 'sigmoid'))
    clf.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics= ['accuracy'])
    return clf

clf = KerasClassifier(build_fn = build_classifier,batch_size=10,epochs=100 )
accuracies = cross_val_score(estimator = clf, X = X_train , y = y_train,cv=10 ,n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()
'''

def build_classifier(optimizer):
   clf = Sequential()

   clf.add(Dense(output_dim = 6,kernel_initializer='uniform', activation = 'relu',input_dim=11))
   clf.add(Dense(output_dim = 6,kernel_initializer='uniform', activation = 'relu'))
   clf.add(Dense(output_dim = 1,kernel_initializer='uniform', activation = 'sigmoid'))
   clf.compile(optimizer = optimizer, loss = 'binary_crossentropy',metrics= ['accuracy'])
   return clf

clf = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25,32] , 'epochs' : [100,500] , 'optimizer' : ['adam','rmsprop']}

grid = GridSearchCV(estimator = clf , param_grid = parameters , scoring = 'accuracy' , cv = 10)
grid = grid.fit(X_train,y_train)
print(grid.best_score_)
print(grid.best_params_)
