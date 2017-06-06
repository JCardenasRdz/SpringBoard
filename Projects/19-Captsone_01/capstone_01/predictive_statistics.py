# quantify the effect of age on Survival
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC

def plot_roc_curve(fpr, tpr, lw = 2, title=''):
    auc = metrics.auc(fpr,tpr);
    plt.figure(figsize =(6,6))
    plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def labels_to_numbers(DataFrame, Variable):
    le = preprocessing.LabelEncoder()
    numbers_ = le.fit_transform(DataFrame[Variable].values)
    return numbers_

def TrainRFC(Xdata,ydata):
    clf= RFC()
    # specify parameters and distributions to sample from
    Forest  = GridSearchCV(clf, param_grid = {"n_estimators": np.arange(10, 100,10),
                                                "max_features": np.arange(1,Xdata.shape[1],1)},
                                                   scoring = make_scorer(cohen_kappa_score),
                                                   verbose = 1, n_jobs = -1);
    Forest.fit(Xdata,ydata);
    return Forest.best_estimator_

def TrainLogRegModel_Kappa(Xdata, ydata):
    clf = LogisticRegression()
    LogRegModel = GridSearchCV(clf, param_grid = {"C": np.arange(1,11,1),
                                                  "fit_intercept": ["True", "False"]},
                                                   scoring = make_scorer(cohen_kappa_score),
                                                   verbose = 0);


    LogRegModel.fit(Xdata,ydata);
    return LogRegModel


def _LogisticRegression(X,y, title =''):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X,y,  train_size=0.50, stratify = y)
    # train
    clf = TrainLogRegModel_Kappa(X_train,y_train);
    pred_prob = clf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, pred_prob);
    kappa = metrics.cohen_kappa_score(clf.predict(X_test),y_test)
    auc =   metrics.auc(fpr,tpr)
    plot_roc_curve(fpr,tpr, title = title)
    return  kappa, auc

def _RFClassifier(X,y, size_train = 0.50):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X,y,  train_size= size_train, stratify = y)
    # train
    clf = TrainRFC(X_train,y_train);
    print(metrics.classification.classification_report(clf.predict(X_test), y_test))
    return clf, X_test, y_test
