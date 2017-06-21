from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics, linear_model
from imblearn import over_sampling
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
import matplotlib.pyplot as plt

RANDOM_STATE = 42; # for reproducibility


def labels_to_numbers(DataFrame, Variable):
    le = preprocessing.LabelEncoder()
    numbers_ = le.fit_transform(DataFrame[Variable].values)
    return numbers_


def binary_classifier_metrics(classifier, Xtrain, Ytrain, Xtest, Ytest):
    # metrics
    predicted_class = classifier.predict(Xtest)

    kappa = metrics.cohen_kappa_score(Ytest,predicted_class)
    np.round(kappa,3)

    auc = metrics.roc_auc_score(Ytest,predicted_class)
    auc = np.round(auc,3)


    # ROC curve
    probability = classifier.predict_proba(Xtest)
    fpr, tpr, _  = metrics.roc_curve(Ytest, probability[:,1], drop_intermediate=False)

    # report
    print(metrics.classification_report(Ytest,predicted_class))
    print('The estimated Cohen kappa is ' + str(kappa))
    print('The estimated AUC is ' + str(auc))
    print('=='*30)
    print('\n'*2)

    return auc, kappa, fpr, tpr


def split_data(Xdata,Ydata, oversample, K_neighbors = 4):
    if oversample == False:
        X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata,
                                                            train_size = 0.70,
                                                            random_state=RANDOM_STATE)
    elif oversample == True:
        print('Data was oversampled using the ADASYN method')
        smote = over_sampling.ADASYN(random_state = RANDOM_STATE, n_neighbors = K_neighbors)
        # split
        X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata,train_size = 0.70, random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_sample(X_train,y_train)


        # oversample the train sets
        #X_over, y_over = smote.fit_sample(Xdata,Ydata)
        #X_train, X_test, y_train, y_test = train_test_split(X_over, y_over,train_size = 0.70,random_state=RANDOM_STATE, stratify = y_over)

    return X_train, X_test, y_train, y_test


# perform Logistic Regression without correcting for unbalance
def Logistic_Regression(Xdata, Ydata, oversample = False, K_neighbors = 4):
    '''
    Perform Logistic Regression optimizing C, penalty, and fit_intercept to maximize
    Cohen kappa (min  = 0, max = 1.0)
    '''

    # split data
    X_train, X_test, y_train, y_test = split_data(Xdata,Ydata, oversample, K_neighbors)

    # train and tune parameters using GridSearchCV
    pars= dict(   C = np.arange(.01,100,.1),
                  penalty = ['l2', 'l1'],
                  fit_intercept = [True,False])

    grid =  GridSearchCV(  linear_model.LogisticRegression(), param_grid = pars,
                           scoring = metrics.make_scorer(metrics.cohen_kappa_score),
                           cv= 5, verbose = 0, n_jobs = -1)

    # fit
    grid.fit(X_train,y_train)

    # metrics
    auc, kappa, fpr, tpr = binary_classifier_metrics(grid, X_train, y_train, X_test, y_test)

    # output
    return auc, kappa, fpr, tpr

# Random Forest Regressor
def RandomForest_Classifier(Xdata, Ydata, oversample = False, K_neighbors = 4, calibrate_prob = True):

    # split data
    X_train, X_test, y_train, y_test = split_data(Xdata,Ydata, oversample, K_neighbors)

    # define parameter grid search
    pars = dict(    n_estimators = np.arange(1,10,1),
                    max_features = np.arange(1, Xdata.shape[1], 1),
                    max_depth = [None, 1, 2, 3, 4, 5])

    # perform grid search
    grid=  GridSearchCV(RFC( random_state = RANDOM_STATE),param_grid = pars,
                           scoring = metrics.make_scorer(metrics.cohen_kappa_score),
                           cv= 3, verbose = 0, n_jobs = -1)

    # fit
    grid.fit(X_train,y_train)

    # get best classifier and calibrate it
    clv = CalibratedClassifierCV(base_estimator = grid.best_estimator_ , method='sigmoid', cv=3)


    # metrics
    auc, kappa, fpr, tpr = binary_classifier_metrics(grid, X_train, y_train, X_test, y_test)

    # output
    return auc, kappa, fpr, tpr, grid.best_estimator_

def plot_forest_feature_importances_(forest, features_legend, title = ''):
    importances = forest.feature_importances_;
    importances = importances / np.max(importances)
    sorted_index = np.argsort(importances)

    x = range(len(importances));

    plt.figure()
    plt.barh(x, importances[sorted_index],color="b", align="center")

    plt.yticks(sorted_index, features_legend);
    plt.title(title);
    plt.xlabel('RELATIVE IMPORTANCE');
    plt.ylabel('PREDICTOR');
    plt.show()


def plot_compare_roc(fpr1_, tpr1_,fpr2_, tpr2_, auc1, auc2, title =''):
    plt.figure()
    plt.plot(fpr1_, tpr1_, fpr2_, tpr2_);
    plt.legend(['Unbalanced | AUC = ' + str(auc1),'Oversampled | AUC = ' + str(auc2)]);
    plt.xlabel('False-positive rate');
    plt.ylabel('True-positive rate');
    plt.title(title);



##  ============== Continous Outcomes  ============== ##

# metrics
mae = metrics.median_absolute_error

def mae_report(Ytest, Yhat, outcome_):
    error = mae(Ytest, Yhat)
    error = np.round( error, decimals=3)
    # report
    print('==' *40)
    print('The median absolute error for testing data set of ' + outcome_ + ' is: ' + str(error))
    print('==' *40)

import seaborn.apionly as sns

def train_test_report(predictor, Xtrain, Ytrain, Xtest, Ytest, outcome):
    # train
    predictor.fit(Xtrain, Ytrain)
    # test
    Yhat = predictor.predict(Xtest)
    # report
    mae_report(Ytest, Yhat, outcome)

    ax = sns.regplot(x = Ytrain, y= predictor.predict(Xtrain));
    ax.set_ylabel('Predicted ' + outcome);
    ax.set_xlabel('Observed ' + outcome);

# lsq
import statsmodels.api as sm
def lsq(Xtrain,Ytrain, Xtest, Ytest, outcome =''):
    # train
    OLS = sm.OLS(Ytrain,Xtrain).fit();
    print(OLS.summary())
    #test
    Yhat = OLS.predict(Xtest)
    # report
    mae_report(Ytest, Yhat, outcome)

# SVR
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# GridSearchCV utility
def gridsearch(regressor, grid):
    optimized_regressor=  GridSearchCV(  regressor,
                               param_grid = grid,
                               cv= 3, verbose = 0, n_jobs = -1,
                               scoring = metrics.make_scorer(metrics.median_absolute_error))

    return optimized_regressor



def svr(Xtrain,Ytrain, Xtest, Ytest, outcome = ''):
    # define regressor
    regressor =  SVR()
    # define parameter grid search
    grid = dict(       kernel = ['rbf','linear','sigmoid'],
                       C = np.arange(1,11,.1),
                       epsilon = np.arange(1,11,.1),
                       gamma = np.linspace(1/10,10,10))
    # perform grid search
    grid_search=  gridsearch(regressor, grid)

    # train, test, and report
    train_test_report(grid_search, Xtrain, Ytrain, Xtest, Ytest, outcome)

    return grid_search


# ElasticNet
from sklearn.linear_model import ElasticNet as ENet

def ElasticNet(Xtrain,Ytrain, Xtest, Ytest, outcome = ''):
    # define regressor
    regressor =  ENet(max_iter=5000)
    # define parameter grid search
    grid = dict(   alpha = np.arange(1,20,.5), l1_ratio = np.arange(.1,1,.05))
    # perform grid search
    grid_search=  gridsearch(regressor, grid)
    # train, test, and report
    train_test_report(grid_search, Xtrain, Ytrain, Xtest, Ytest, outcome)

    return grid_search

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor as RFR

def RandomForestRegressor(Xtrain,Ytrain, Xtest, Ytest, outcome = ''):
    # define regressor
    regressor =  RFR( criterion='mse', random_state = RANDOM_STATE)

    #
    num_features = Xtrain.shape[1]

    # define parameter grid search
    grid = dict(    n_estimators = np.arange(5,100,5),
                    max_features = np.arange(1,num_features, 1),
                    max_depth = [None, 1, 2, 3, 4, 5])

    # perform grid search
    grid_search=  gridsearch(regressor, grid)

    # train, test, and report
    train_test_report(grid_search, Xtrain, Ytrain, Xtest, Ytest, outcome)

    return grid_search
