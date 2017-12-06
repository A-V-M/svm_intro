# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 04:14:10 2017

@author: andreas
"""
from sklearn.metrics import accuracy_score, recall_score,precision_score, make_scorer,fbeta_score
from sklearn import svm, tree
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, ShuffleSplit,train_test_split
import matplotlib.pyplot as plt

def classhow(features,y,params):
    
    n_samples = len(y)   
    n_samples_per_category = int(n_samples/2)
    accuracy = dict()

    X_train, X_test, y_train, y_test = train_test_split(features.T, y, test_size = 0.2, random_state = 0)    

    clf_SVM = svm.SVC(**params,random_state=0)

    clf_fit = clf_SVM.fit(X_train,y_train)
    
    preds = clf_fit.predict(X_test)

    accuracy["test"] = float(accuracy_score(preds,y_test))
    
    preds = clf_fit.predict(X_train)
    
    accuracy["train"] = float(accuracy_score(preds,y_train))

    return accuracy, clf_fit

y = [0] * 100 + [1] * 100

n_samples = len(y)   
n_samples_per_category = int(n_samples/2)

mean1 = [-1,1]
mean2 = [1,-1]
    
cov1 = [[2,0.1],[0.1,2]]
cov2 = [[2,0.1],[0.1,2]]
    
data = np.vstack([np.random.multivariate_normal(mean1, cov1 ,n_samples_per_category),
           np.random.multivariate_normal(mean2, cov2, n_samples_per_category)])
              
scenario = {"Mean 1": mean1,"Covariance matrix 1": cov1, 
            "Mean 2": mean2,"Covariance matrix 2": cov2,
            "Data": data
           }

features = scenario["Data"].T    
features = (features.T - np.mean(features,axis=1)).T

f, ax = plt.subplots(3,2)
ax = ax.reshape(1,6)[0]
acc = [0] * 6

for i,vals in enumerate([0.001,0.01,0.1,1,10,100]):

    params = {"kernel": 'linear', "C": vals}
   
    acc[i], clf_fit = classhow(features,y,params)
    ax[i].plot(features[0,0:n_samples_per_category],features[1,0:n_samples_per_category], '.r')
    ax[i].plot(features[0,n_samples_per_category:n_samples],features[1,n_samples_per_category:n_samples], '.b')
               
    xlim = ax[i].get_xlim()
    ylim = ax[i].get_ylim()

    XX, YY = np.mgrid[xlim[0]:xlim[1]:200j, ylim[0]:ylim[1]:200j]
    Z = clf_fit.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    ax[i].contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],levels=[-.5, 0, .5])
    ax[i].set_title([params,acc[i]])


