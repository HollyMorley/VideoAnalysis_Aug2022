import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy

def compute_regression(X, y, raw_features, cv=5):
    model = LogisticRegression(penalty='none', fit_intercept=False)
    model.fit(X.T, y)
    w = model.coef_

    y_pred = np.dot(w, X)
    # change y_pred +ves to 1 and -ves to 0
    y_pred[y_pred > 0] = 1
    y_pred[y_pred < 0] = 0

    bal_acc = balanced_accuracy(y.T, y_pred.T)

    return w, bal_acc

