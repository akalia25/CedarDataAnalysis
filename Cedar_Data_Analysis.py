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
    ax = sns.distplot(df.amount_due_after_insurance, rug=True,
                      hist=False, axlabel='Amount Due After Insurance')
    feature_cols = ['amount_outstanding' , 'num_insurers' , 'has_ecomms',
                    'current_engagements', 'median_household_income']
    X = df[feature_cols]
    y = df['any_payment_made_within_120']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train,y_train)
    y_pred=logreg.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))

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

    selector = RFE(logreg, n_features_to_select = 1)
    selector = selector.fit(X_train, y_train)
    order = selector.ranking_
    feature_ranks = []
    for i in order:
        feature_ranks.append(f"{i}. {feature_cols[i-1]}")
    print(feature_ranks)
def main():
    df = uploadfile()
    print(df.head())


if __name__ == '__main__':
    main()
