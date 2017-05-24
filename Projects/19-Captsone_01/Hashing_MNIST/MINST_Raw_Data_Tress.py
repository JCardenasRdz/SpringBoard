
# coding: utf-8

# # Classification of MNIST dataset digits
# ## Labels = 10
# ## Training Samples =  60,000
# ## Testing Samples =   10,000

# In[1]:

# Modules
import numpy as np
from keras.datasets import mnist
from sklearn import preprocessing
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt

 # for reproducibility
np.random.seed(123)

# get data from Keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# normalize
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# encode outputs
le = preprocessing.LabelEncoder().fit(y_train)
Y_train = le.transform(y_train)
Y_test = le.transform(y_test)


# In[4]:
# prepare data for classification
# Note: I am transforming each 28*28 image into a 1 X 784 vector
X_train_matrix = np.zeros( (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
X_test_matrix = np.zeros( (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

for idx, I in enumerate(X_train):
    X_train_matrix[idx,:] = I.reshape(1,28*28)

for idx, I in enumerate(X_test):
    X_test_matrix[idx,:] = I.reshape(1,28*28)


# ### Decision Tree Classifier
clasifier = tree.DecisionTreeClassifier()
clasifier = clasifier.fit(X_train_matrix, Y_train)
y_pred = clasifier.predict(X_test_matrix)

print("Classification Report:\n%s" % metrics.classification_report(Y_test, y_pred))


# ### RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=10)
RFC = RFC.fit(X_train_matrix, Y_train)
y_pred = RFC.predict(X_test_matrix)


# In[8]:

print("Classification Report:\n%s" % metrics.classification_report(Y_test, y_pred))
