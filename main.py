#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import locale
locale.setlocale(locale.LC_ALL, 'turkish')
import time
from sklearn.utils import shuffle
#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC


import copy
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_curve, roc_auc_score
import torch

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier

from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# used for normalization
from sklearn.preprocessing import StandardScaler

# used for cross-validation
from sklearn.model_selection import StratifiedKFold

# used to compute accuracy
from sklearn.metrics import accuracy_score

# this is an incredibly useful function
from pandas import read_csv

# local functions
import genericFunctions
time.strftime('%c')



# here the feature file is selected
featureFile = "feature-importance-efs_father_learning_problems.csv"

# load dataset
X, y, featureNames = genericFunctions.loadTCGADataset()
print("Training dataset (original):", X.shape)

# load selected features
selectedFeatures = genericFunctions.loadFeatures(featureFile)

# create reduced dataset
print("Reading feature file \"" + featureFile + "\"...")
featureIndexes = [i for i in range(0, len(featureNames)) if featureNames[i] in selectedFeatures]
featureIndexes
X_reduced = X[:, featureIndexes]
print("Training dataset (reduced):", X_reduced.shape)

print("Normalizing by samples...")
normalizeBySample = True
if normalizeBySample:
    from sklearn.preprocessing import normalize
    X = normalize(X)
    X_reduced = normalize(X_reduced)
    y=y.reshape((len(y),1))
veriler_read=pd.DataFrame.from_records(np.concatenate((X_reduced, y), axis=1))


oran =int((len(veriler_read) / 3))

veriler_read= shuffle(veriler_read)
veriler_test=veriler_read.head(oran)
aaaa=~(veriler_read.isin(veriler_test))
veriler_train =veriler_read[~veriler_read.isin(veriler_test)]
veriler_train=veriler_train[veriler_train[0].notna()]
veriler_test=veriler_test.reset_index(drop=True)
veriler_train =veriler_train.reset_index(drop=True)
print (len(veriler_test))
print(len(veriler_train))
df_logprice =veriler_train.iloc[:,50:51]
df_features = veriler_train.iloc[:,0:50]
features = veriler_train.iloc[:,0:50].values
logprice =veriler_train.iloc[:,50:51].values
x_train1=df_features
y_train1=df_logprice
train11 = pd.concat([x_train1, y_train1], axis=1)
df_logprice_test =veriler_test.iloc[:,50:51]
df_features_test = veriler_test.iloc[:,0:50]
features_test = veriler_test.iloc[:,0:50].values
logprice_test =veriler_test.iloc[:,50:51].values
x_test1=df_features_test
y_test1=df_logprice_test
print(x_test1.head(2))
whole_dataset = pd.concat([train11, x_test1], axis=0)
xxxx=whole_dataset.corr(method ='pearson')
from sklearn.linear_model import LinearRegression

#lin_reg = LinearRegression()
svc=SVC(kernel='linear',probability=True)

#lin_reg.fit(x_train1, y_train1)
svc.fit(x_train1, y_train1)
#lin_reg_tahmin = lin_reg.predict(x_train1)
svc_tahmin = svc.predict(x_train1)
#y_error = y_train1.values - lin_reg_tahmin
y_error=svc.predict_proba(x_train1)[:, 1]
#MAE = np.mean(np.abs(y_train1 - svc.predict_proba(x_train1)[:, 1]), axis=0)
df_y_error= pd.DataFrame( data = y_error, index= range(len(y_error)), columns=["y_error"])
veriler_train2 =pd.concat([veriler_train,df_y_error],axis=1)
print(veriler_train2.head(2))
print(len(veriler_train2))

wcss=[]
for i in range(1,10):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=i, random_state=0).fit(df_y_error)
    kmeans.labels_
    #print(kmeans.cluster_centers_)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,10),wcss)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_y_error)
kmeans.labels_
#print(kmeans.cluster_centers_)
#veriler.append(kmeans.inertia_)
veriler_train2['cluster'] = kmeans.labels_
veriler_train2.head(2)
print(veriler_train2.groupby('cluster')['cluster'].count())

df_logprice_s =veriler_train2.iloc[:,52:53]
df_features_s = veriler_train2.iloc[:,0:50]
features_s = veriler_train2.iloc[:,0:50].values
logprice_s =veriler_train2.iloc[:,52:53].values
x_train_s=df_features_s
y_train_s=df_logprice_s

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(x_train_s,y_train_s)
knn_y_pred = knn.predict(x_test1)

df_knn_y_pred= pd.DataFrame( data = knn_y_pred, index= range(len(knn_y_pred)), columns=["cluster"])
df_knn_test=pd.concat([x_test1,df_knn_y_pred],axis=1)


veriler_all=pd.concat([veriler_train2,df_knn_test],axis=0,ignore_index=True)
veriler=veriler_all.drop(['y_error'], axis=1)
#veriler = veriler_all[['logm2', 'nroom','nbathr_tr', 'floor','nfloor', 'insite','year', 'logprice','cluster']]
veriler.head(2)

veriler.groupby('cluster')['cluster'].count()

import math as math
def mape(actual, predict):
    tmp, n = 0.0, 0
    for i in range(0, len(actual)):
        if actual[i] != 0:
            tmp += math.fabs(actual[i]-predict[i])/actual[i]
            n += 1
    return float((tmp/n)*100)


# Cluster version
import warnings

warnings.filterwarnings("ignore")
# from sklearn.utils import shuffle
from sklearn.model_selection import KFold  # import KFold

# veriler2= veriler.head(100)
veriler2 = veriler.sort_values(by=['cluster'])
cluster_list = veriler.cluster.unique()

fold1 = 0
fold2 = 0

df_mse1 = pd.DataFrame(
    columns=['fold1', 'fold2', 'i', 'c', 'gamma', 'nu', 'mse', 'rmse', 'mae', 'mape', 'r2', 'adjusted_r2_square'])
for i in range(0, len(cluster_list)):
    df_mse2 = pd.DataFrame(columns=['fold1', 'fold2', 'i', 'c', 'gamma', 'nu', 'mse'])
    kfold_cluster = []
    for cls_veriler in veriler2.values:
        if i in cls_veriler[51:52]:
            kfold_cluster.append(cls_veriler)
    X1 = np.asarray(kfold_cluster)
    kf1 = KFold(5, True, 1)
    for train1, test1 in kf1.split(X1):
        fold1 += 1
        train1 = X1[train1]
        test1 = X1[test1]
        #df_train1 = pd.DataFrame(data=train1, index=range(len(train1)),
        #                        columns=["logm2", "nroom", "nbathr_tr", "floor", "nfloor", "insite", "year",
        #                                  "logprice", "cluster"])
        train1= train1[~np.isnan(train1).any(axis=1)]
        df_train1= pd.DataFrame(train1)
        #df_test1 = pd.DataFrame(data=test1, index=range(len(test1)),
        #                        columns=["logm2", "nroom", "nbathr_tr", "floor", "nfloor", "insite", "year", "logprice",
        #                                 "cluster"])
        df_test1=pd.DataFrame(test1)
        y_test1 = df_test1.iloc[:, 50:51]
        x_test1 = df_test1.iloc[:, 0:50]
        y_train1 = df_train1.iloc[:, 50:51]
        x_train1 = df_train1.iloc[:, 0:50]

        X2 = df_train1.values
        kf2 = KFold(5, True, 1)
        for train2, test2 in kf2.split(X2):
            fold2 += 1
            train2 = X2[train2]
            test2 = X2[test2]
            df_train2 = pd.DataFrame(train2)
            df_test2 = pd.DataFrame(test2)

            y_test2 = df_test2.iloc[:, 50:51]
            x_test2 = df_test2.iloc[:, 0:50]
            y_train2 = df_train2.iloc[:, 50:51]
            x_train2 = df_train2.iloc[:, 0:50]


#######################################
            # #Setting values for the parameters
            # #n_estimators = [100, 300, 500, 800, 1200]
            # max_depth = [5, 10, 15, 25, 30]
            # min_samples_split = [2, 5, 10, 15, 100]
            # min_samples_leaf = [1, 2, 5, 10]
            # max_features = [1, 2, 5, 10]
            #
            # #Creating a dictionary for the hyper parameters
            # hyperT = dict(max_depth = max_depth, min_samples_split = min_samples_split,
            #               min_samples_leaf = min_samples_leaf, max_features=max_features)
            # tree = DecisionTreeClassifier()
            # #Applying GridSearchCV to get the best value for hyperparameters
            # gridT = GridSearchCV(tree, hyperT, cv = 3, verbose = 1, n_jobs = -1)
            # bestTree = gridT.fit(x_train2, y_train2)
            # print('The best hyper parameters are: \n',gridT.best_params_)
            # #Fitting the decision tree model with the best hyper parameters obtained through GridSearchCV
            #
            # pred_tree = bestTree.predict(x_test2)
            #
            # #Checking different metrics for decision tree model after tuning the hyperparameters
            # print('Checking different metrics for decision tree model after tuning the hyperparameters:\n')
            # print("Training accuracy: ",bestTree.score(x_train2,y_train2))
            # acc_score = accuracy_score(y_test2, pred_tree)
            # print('Testing accuracy: ',acc_score)
            # conf_mat = confusion_matrix(y_test2, pred_tree)
            # print('Confusion Matrix: \n',conf_mat)
            # roc_auc = roc_auc_score(y_test2,pred_tree)
            # print('ROC AUC score: ',roc_auc)
            # class_rep2 = classification_report(y_test2,pred_tree)
            # print('Classification Report: \n',class_rep2)


            ###########################################
           #  # Setting values for the parameters
           #  n_estimators = [100, 300, 500, 800, 1200]
           #  # max_depth = [5, 10, 15, 25, 30]
           #  max_samples = [5, 10, 25, 50, 100]
           #  max_features = [1, 2, 5, 10, 13]
           #  bagg = BaggingClassifier()
           #  # Creating a dictionary for the hyper parameters
           #  hyperbag = dict(n_estimators=n_estimators, max_samples=max_samples,
           #                  max_features=max_features)
           #
           #  # Applying GridSearchCV to get the best value for hyperparameters
           #
           #
           #
           #
           #
           #  gridbag = GridSearchCV(bagg, hyperbag, cv=3, verbose=1, n_jobs=-1)
           #  bestbag = gridbag.fit(x_train2, y_train2)
           #  pred_bagg1 = bestbag.predict(x_test2)
           #  # Checking different metrics for bagging model after tuning the hyperparameters
           #  print('Checking different metrics for bagging model after tuning the hyperparameters:\n')
           #  print("Training accuracy: ", bestbag.score(x_train2, y_train2))
           #  acc_score = accuracy_score(y_test2, pred_bagg1)
           #  print('Testing accuracy: ', acc_score)
           #  conf_mat = confusion_matrix(y_test2, pred_bagg1)
           #  print('Confusion Matrix: \n', conf_mat)
           #  #roc_auc = roc_auc_score(y_test1, pred_bagg1)
           # # print('ROC AUC score: ', roc_auc)
           #  class_rep2 = classification_report(y_test2, pred_bagg1)
           #  print('Classification Report: \n', class_rep2)


           #################################
        # rf = RandomForestClassifier()
        # # Setting values for the parameters
        # n_estimators = [100, 300, 500, 800, 1200]
        # max_depth = [5, 10, 15, 25, 30]
        # min_samples_split = [2, 5, 10, 15, 100]
        # min_samples_leaf = [1, 2, 5, 10]
        #
        # # Creating a dictionary for the hyper parameters
        # hyper_rf = dict(n_estimators=n_estimators, max_depth=max_depth,
        #                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        #
        # # Applying GridSearchCV to get the best value for hyperparameters
        # gridrf = GridSearchCV(rf, hyper_rf, cv=3, verbose=1, n_jobs=-1)
        # bestrf = gridrf.fit(x_train2, y_train2)
        # # Checking different metrics for random forest model after tuning the hyperparameters
        # print('Checking different metrics for random forest model after tuning the hyperparameters:\n')
        # print("Training accuracy: ", gridrf.score(x_train2, y_train2))
        # acc_score = accuracy_score(y_test2, bestrf)
        # print('Testing accuracy: ', acc_score)
        # conf_mat = confusion_matrix(y_test2, bestrf)
        # print('Confusion Matrix: \n', conf_mat)
        # roc_auc = roc_auc_score(y_test2, bestrf)
        # print('ROC AUC score: ', roc_auc)
        # class_rep3 = classification_report(y_test2, bestrf)
        # print('Classification Report: \n', class_rep3)

        # from sklearn.ensemble import GradientBoostingClassifier
        # from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        # p_test3 = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750]}
        #
        # tuning = GridSearchCV(estimator =GradientBoostingClassifier(max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10),
        #             param_grid = p_test3, scoring='accuracy',n_jobs=4, cv=5)
        # bestrf =tuning.fit(x_train2, y_train2.values.ravel())
        # predictions = bestrf.predict(x_test2)
        #
        # print("Confusion Matrix:")
        # print(confusion_matrix(y_test2, predictions))
        # print()
        # print("Classification Report")
        # print(classification_report(y_test2, predictions))
        # y_scores_gb = tuning.decision_function(x_test2)
        # fpr_gb, tpr_gb, _ = roc_curve(y_test2, y_scores_gb)
        # roc_auc_gb = auc(fpr_gb, tpr_gb)
        #
        # print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))

        # from sklearn.linear_model import RidgeClassifier
        # from sklearn.datasets import make_blobs
        # from sklearn.model_selection import RepeatedStratifiedKFold
        # from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        # # define dataset
        # X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
        # # define models and parameters
        # model = RidgeClassifier()
        # alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # # define grid search
        # grid = dict(alpha=alpha)
        # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',
        #                            error_score=0)
        # bestrf = grid_search.fit(x_train2, y_train2.values.ravel())
        # predictions = bestrf.predict(x_test2)
        #
        # print("Confusion Matrix:")
        # print(confusion_matrix(y_test2, predictions))
        # print()
        # print("Classification Report")
        # print(classification_report(y_test2, predictions))
        # y_scores_gb = grid_search.decision_function(x_test2)
        # fpr_gb, tpr_gb, _ = roc_curve(y_test2, y_scores_gb)
        # roc_auc_gb = auc(fpr_gb, tpr_gb)
        #
        # print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))

        from sklearn.linear_model import SGDClassifier
        from sklearn.datasets import make_blobs

        # from sklearn.model_selection import RepeatedStratifiedKFold
        # from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        # # define dataset
        # X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
        # # define models and parameters
        # model = SGDClassifier()
        #
        # # define grid search
        # grid = param_grid = {
        #     'loss': ['log'],
        #     'penalty': ['elasticnet'],
        #     'alpha': [10 ** x for x in range(-6, 1)],
        #     'l1_ratio': [0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1],
        # }
        # clf = SGDClassifier(random_state=0, class_weight='balanced')
        #
        # grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
        #                             n_jobs=-1, scoring='roc_auc')
        # bestrf = grid_search.fit(x_train2, y_train2.values.ravel())
        # predictions = bestrf.predict(x_test2)
        #
        # print("Confusion Matrix:")
        # print(confusion_matrix(y_test2, predictions))
        # print()
        # print("Classification Report")
        # print(classification_report(y_test2, predictions))
        # y_scores_gb = grid_search.decision_function(x_test2)
        # fpr_gb, tpr_gb, _ = roc_curve(y_test2, y_scores_gb)
        # roc_auc_gb = auc(fpr_gb, tpr_gb)
        #
        # print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))

        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['linear']}

        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

        # fitting the model for grid search
        grid.fit(x_train2, y_train2.values.ravel())

        grid_predictions = grid.predict(x_test2)

        # print classification report
        print(classification_report(y_test2, grid_predictions))

        predictions =grid_predictions

        print("Confusion Matrix:")
        print(confusion_matrix(y_test2, predictions))
        print()
        print("Classification Report")
        print(classification_report(y_test2, predictions))
        y_scores_gb = grid.decision_function(x_test2)
        fpr_gb, tpr_gb, _ = roc_curve(y_test2, y_scores_gb)
        roc_auc_gb = auc(fpr_gb, tpr_gb)

        print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))

