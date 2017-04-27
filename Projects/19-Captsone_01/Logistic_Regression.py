# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# allocate and train
model2 = LogisticRegression()
model2.fit(X_train, y_train)

# predict outcomes for the test set
predicted = model2.predict(X_test)

# calculate class probabilities
probs = model2.predict_proba(X_test)

# generate evaluation metrics
# accurracy
print(20*"--")
print('The accuracy is: ')
print(metrics.accuracy_score(y_test, predicted)*100)

# AUC
print(20*"--")
print('The AUC is: ')
print(metrics.roc_auc_score(y_test, probs[:, 1]))

# confusion matrix
print(20*"--")
print('The confusion matrix is: ')
print(metrics.confusion_matrix(y_test, predicted))

# evaluate the model using leave-one-out  cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print(20*"--")
print('The leave-one-out  accuracy is: ')
print(scores.mean()*100)
