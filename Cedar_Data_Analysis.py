#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:58:03 2019

@author: adityakalia
"""
import pandas as pd


def uploadfile():
    df = pd.read_csv("/Users/adityakalia/Documents/Cedar/analytics_challenge_dataset.csv")
    return df


def main():
    df = uploadfile()
    print(df.head())


if __name__ == '__main__':
    main()
