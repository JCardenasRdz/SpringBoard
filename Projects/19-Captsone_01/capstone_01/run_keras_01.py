from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

import numpy as np
from dataloader import *

np.set_printoptions(threshold = np.nan)
trainX, trainY, testX, testY = import_data()

num_features = trainX.shape[1]
num_labels = trainY.shape[1]

model = Sequential()

model.add( Dense(num_labels, input_dim = num_features, activation = 'softmax'))

num_epochs = 3000

sgd = optimizers.SGD(lr = 0.0008, decay = 1e-6, momentum=0.9, nesterov=True)
# compile
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

#
history = model.fit(trainX, trainY, batch_size = 128,
                    nb_epoch = num_epochs, verbose = 1, validation_split= 0.3)

score = model.evaluate(testX, testY, verbose = 1)
print('Test score:',score[0])
print('Test accurracy:',score[1])
