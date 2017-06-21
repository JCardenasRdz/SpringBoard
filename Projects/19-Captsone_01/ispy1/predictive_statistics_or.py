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

def labels_to_numbers(DataFrame, Variable):
    le = preprocessing.LabelEncoder()
    numbers_ = le.fit_transform(DataFrame[Variable].values)
    return numbers_

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

# Plot the feature importances of the forest
def Tree_feature_importances(Forest):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f): " % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection as ms
from imblearn import pipeline as pl
from sklearn.model_selection import train_test_split


def validation_curve(Classifier, X, y,parameter_to_optimize, scorer, parameter_range = np.arange(1,5,1), c_v = 3):
    train_scores, test_scores = ms.validation_curve(
                                                 Classifier,
                                                  X, y,
                                                   param_name = parameter_to_optimize, param_range = parameter_range,
                                                    cv= c_v, scoring = scorer, n_jobs=1)

    idx = np.argmax(np.median(test_scores, axis = 1))


    return train_scores, test_scores, parameter_range[idx]

def plot_with_errors(ydata):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(param_range, test_scores_mean, label='mean of metric')
    ax.fill_between(param_range, test_scores_mean + test_scores_std,
                test_scores_mean - test_scores_std, alpha=0.2)
    plt.show()

def plot_validation_curve(train_scores, test_scores, param_range, xlabel='x', ylabel='y', title =''):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(param_range, test_scores_mean, label='mean')
    ax.fill_between(param_range, test_scores_mean + test_scores_std,
                test_scores_mean - test_scores_std, alpha=0.2)

    idx_max = np.argmax(np.mean(test_scores, axis=1))

    plt.scatter(param_range[idx_max], test_scores_mean[idx_max],
            label=r'Cohen Kappa: ${0:.2f}\pm{1:.2f}$'.format(
                test_scores_mean[idx_max], test_scores_std[idx_max]))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    #plt.xlim([1, 10])
    #plt.ylim([0.4, 0.8])

    plt.legend(loc="best")
    plt.show()

def classification_report(y_expected, yhat):
    #  test performance
    print(20 * '---')
    print('Observed Performance')
    print(20 * '---')
    print(metrics.classification_report(y_expected, yhat))


    index_largest_class = np.argmax(pd.Series(y_expected).value_counts().values)
    index_smallest_class = np.argmin(pd.Series(y_expected).value_counts().values)
    largest_class = pd.Series(y_expected).value_counts().index[index_largest_class]
    small_class = pd.Series(y_expected).value_counts().index[index_smallest_class]

    y_hat_crazy = np.zeros_like(yhat)
    y_hat_crazy[:] = largest_class
    y_hat_crazy[0] = small_class
    size = y_hat_crazy.shape[0] - 1

    # How would this look if I predict everything belong to the largest class?
    print(20 * '---')
    print('Performance assuming '+' '+str(size)+' observations belong to the largest class')
    print(20 * '---')
    print(metrics.classification_report(y_expected, y_hat_crazy))
