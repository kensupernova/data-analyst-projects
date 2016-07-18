#!/usr/bin/python

"""
financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
 'poi', 'shared_receipt_with_poi']

POI label: ['poi'] (boolean, represented as integer)
"""

#
# all_features = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
# 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
# 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
#                 'to_messages',
#                 'from_poi_to_this_person', 'from_messages',
#                 'from_this_person_to_poi', 'shared_receipt_with_poi']
all_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                'to_messages',
                'from_poi_to_this_person', 'from_messages',
                'from_this_person_to_poi', 'shared_receipt_with_poi']

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

##########################################################################
n_count =  len(data_dict)
print "total number of data points: "+ str(n_count)
###########################################################################
print "number of features: " + str(len(data_dict["TOTAL"]))

n_poi = 0
for key, value in data_dict.items():
    # print value['poi']
    if value['poi']:
        n_poi = n_poi+1
print n_poi
print len(data_dict) - n_poi

##### investigate the ratio of NAN in each feature

values = data_dict.values()
print values
print data_dict["TOTAL"].keys()

# nan_features = ['deferral_payments', 'loan_advances', 'restricted_stock_deferred', 'deferred_income', 'director_fees']
# for i, feature in enumerate(nan_features):
#     print feature
#     nan_row = []
#     nan_count = 0
#     for key, value in data_dict.items():
#          # print value
#          # print value[feature]
#          nan_row.extend([value[feature]])
#          if value[feature] == "NaN":
#              nan_count = nan_count +1
#     print nan_row
#     # print nan_count
#     print nan_count/146.00


feature_nan_ratio = []
for feature in all_features:
    print feature
    nan_count = 0
    for key, value in data_dict.items():
        feature_value = value[feature]
        # feature_value_upper = feature_value.upper()
        if feature_value == 'NaN':
            nan_count = nan_count + 1
    n_count = float(n_count)
    nan_count = float(nan_count)
    print nan_count
    print nan_count/n_count
    feature_nan_ratio.append((feature, nan_count/n_count))

import pandas as pd
feature_nan_ratio.sort(key=lambda x: x[1], reverse=True)
print feature_nan_ratio

feature_nan_ratio_df = pd.DataFrame(data=feature_nan_ratio)
feature_nan_ratio_df.columns = ["feature", "nan ratio"]
print feature_nan_ratio_df
feature_nan_ratio_df.to_csv("p5_nan_ratio.csv")

