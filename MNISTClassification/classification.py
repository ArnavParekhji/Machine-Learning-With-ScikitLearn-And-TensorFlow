import pandas as pd
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

f = open("mnist.pickle", "rb")
mnist = pickle.load(f)

X, y = mnist["data"], mnist["target"]


##-------VIEW AN IMAGE IN THE DATASET-------

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
##plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
##plt.axis("off")
##plt.show()

##------------------------------------------

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

print(sgd_clf.predict([some_digit]))

##-------MEASURING ACCURACY BY IMPLEMENTING CROSS-VALIDATION-------

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

##skfolds = StratifiedKFold(n_splits=3, random_state=42)

##for train_index, test_index in skfolds.split(X_train, y_train_5):
##    clone_clf = clone(sgd_clf)
##    X_train_folds = X_train[train_index]
##    y_train_folds = (y_train_5[train_index])
##    X_test_fold = X_train[test_index]
##    y_test_fold = (y_train_5[test_index])
##
##    clone_clf.fit(X_train_folds, y_train_folds)
##    y_pred = clone_clf.predict(X_test_fold)
##    n_correct = sum(y_pred == y_test_fold)
##    print(n_correct / len(y_pred))

##-----------------------------------------------------------------

##-------USING CROSS-VALIDATION-------

from sklearn.model_selection import cross_val_score

##print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

##------------------------------------

##-------USING A CONFUSION MATRIX-------

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

##print(confusion_matrix(y_train_5, y_train_pred))

##--------------------------------------

##-------PRECISION AND RECALL-------

from sklearn.metrics import precision_score, recall_score

##print(precision_score(y_train_5, y_train_pred))
##print(recall_score(y_train_5, y_train_pred))

##-------Find f1_score = 2 / ((1 / precision) + (1 / recall))

from sklearn.metrics import f1_score

print(f1_score(y_train_5, y_train_pred))

##----------------------------------

