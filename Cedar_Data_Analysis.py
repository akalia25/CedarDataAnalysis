#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:58:03 2019

@author: adityakalia
"""
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics

def uploadfile():
    df = pd.read_csv("/Users/adityakalia/Documents/CedarDataAnalysis/analytics_challenge_dataset.csv")
    return df

def analysis(df):
    dfMean = df.mean()
    dfMedian = df.median()
    dfStd = df.std()
    df = df.fillna('0')
    patient_max_age = []
    for row in df['patient_age_bucket']:
        if len(row) > 1:
            patient_max_age.append(row[-2:])
        else:
            patient_max_age.append(0)
    df['patient_max_age'] = patient_max_age
    logreg = LogisticRegression()
    cols = df.columns.values.tolist()
    cols.remove('patient_age_bucket')
    cols.remove('any_payment_made_within_120')
    X = df[cols]
    y = df['any_payment_made_within_120']
    X_train,X_test,y_train, y_test=train_test_split(X, y, test_size=0.20,random_state=0)
    logreg.fit(X_train, y_train)
    y_pred=logreg.predict(X_test)
    selector = RFE(logreg, n_features_to_select=15)
    selector = selector.fit(X_train, y_train)
    order = selector.ranking_
    feature_ranks = []
    for idx, item in enumerate(selector.ranking_):
        feature_ranks.append(cols[idx] + ' rank: ' + str(item))
    feature_cols = []
    for idx, item in enumerate(selector.support_):
        if item == True:
            feature_cols.append(cols[idx])
    X = df[feature_cols]
    y = df['any_payment_made_within_120']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
    logreg.fit(X_train,y_train)
    y_pred=logreg.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    feature_importance = abs(logreg.coef_[0])
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    featfig = plt.figure()
    featax = featfig.add_subplot(1, 1, 1)
    featax.barh(pos, feature_importance[sorted_idx], align='center')
    featax.set_yticks(pos)
    featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=8)
    featax.set_xlabel('Relative Feature Importance')
    plt.tight_layout()
    plt.show()

def main():
    df = uploadfile()
    analysis(df)

if __name__ == '__main__':
    main()
