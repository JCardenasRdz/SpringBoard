import pandas as pd
file = 'I-SPY_1_All_Patient_Clinical_and_Outcome_Data.xlsx'

# load and set index of predictors
outcomes_df = pd.read_excel(file, sheetname='outcomes')
outcomes_df = outcomes_df.set_index('SUBJECTID')

# drop NaN
ISPY.dropna(inplace=True)

# drop Columns I don't need
ISPY.drop(['DataExtractDt','Her2MostPos','HR_HER2_CATEGORY','HR_HER2_STATUS'],
                                                        axis=1,inplace=True)
#encode race and drop initial variable
ISPY = ISPY.join(pd.get_dummies(ISPY['race_id'], prefix=['Race']))
ISPY.drop(['race_id'], axis=1,inplace=True)

# load and set index of predictors
outcomes_df = pd.read_excel(file, sheetname='outcomes')
outcomes_df = outcomes_df.set_index('SUBJECTID')\n",
    "\n",
    "#merge PCR and predictors using the Subject ID index\n",
    "df = ISPY.join(outcomes_df['PCR'])\n",

    "# drop NaN\n",
    "df.dropna(inplace=True)"

    "ISPY.drop(['race_id'], axis=1,inplace=True)"



"#Logistic Regression\n",
"# Modules\n",
"from sklearn.linear_model import LogisticRegression\n",
"from sklearn.cross_validation import train_test_split\n",
"from sklearn import metrics\n",
"from sklearn.cross_validation import cross_val_score\n",
"\n",
"# define X and Y\n",
"X = df.iloc[:,0:15].values\n",
"y = df.iloc[:,-1].values\n",
"\n",
"# instantiate a logistic regression model, and fit with X and y\n",
"model = LogisticRegression()\n",
"model = model.fit(X, y)\n",
"\n",
"# check the accuracy without cross validation\n",
"model.score(X, y)"



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
