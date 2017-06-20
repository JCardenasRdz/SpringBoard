from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

import numpy as np
from dataloader import *

np.set_printoptions(threshold = np.nan)
trainX, trainY, testX, testY = import_data()

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

clf = RandomForestClassifier( n_estimators=20)
clf.fit (trainX, trainY)

yhat = clf.predict(testX)
