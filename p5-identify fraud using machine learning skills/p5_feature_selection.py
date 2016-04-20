#!/usr/bin/python

"""
financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
 'poi', 'shared_receipt_with_poi']

POI label: ['poi'] (boolean, represented as integer)
"""


all_features = ['poi','salary', 'deferral_payments',
                'total_payments',
                'loan_advances', 'bonus',
                'restricted_stock_deferred', 'deferred_income',
                'total_stock_value', 'expenses',
                'exercised_stock_options',
                'other', 'long_term_incentive',
                'restricted_stock', 'director_fees',
                'to_messages',
                'from_poi_to_this_person', 'from_messages',
                'from_this_person_to_poi', 'shared_receipt_with_poi',
                "from_ratio", "to_ratio"]


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

### Task 2: Remove outliers
names =  data_dict.keys()
data_dict.pop("TOTAL")
# data_dict.pop("LAY KENNETH L")
# data_dict.pop("BHATNAGAR SANJAY")


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = data_dict
# create a new variable get the ratio of from_this_person_to_poi / from_messages
for key, value in data_dict.iteritems():
    # print my_dataset[key]
    if data_dict[key]["from_this_person_to_poi"] != "NaN" and data_dict[key]["from_messages"] != "NaN":
        my_dataset[key]["from_ratio"] = round(float(data_dict[key]["from_this_person_to_poi"])/data_dict[key]["from_messages"], 5)
    else:
        my_dataset[key]["from_ratio"] = "NaN"

# create a new variable get the ratio of from_poi_to_this_person / to_messages
for key, value in data_dict.iteritems():
    # print my_dataset[key]
    if data_dict[key]["from_poi_to_this_person"] != "NaN" and data_dict[key]["to_messages"] != "NaN":
        my_dataset[key]["to_ratio"] = round(float(data_dict[key]["from_poi_to_this_person"])/data_dict[key]["to_messages"], 5)
    else:
        my_dataset[key]["to_ratio"] = "NaN"


##################################################################
#### split features and labels
select_k = []

for k_i in range(5, 22, 1):
    data = featureFormat(my_dataset, all_features, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.4, random_state=42)

    ##### scale the features
    # minmax scale features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler().fit(features_train)
    X_train_std = scaler.transform(features_train)
    X_test_std = scaler.transform(features_test)

    ###################################################################
    ###### select features
    ### using kbest
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    select = SelectKBest(chi2, k=k_i)
    X_train_std_new = select.fit_transform(X_train_std, labels_train)
    selected= select.get_support()

    kbest_features= ['poi']
    for i, item in enumerate(selected):
        if item:
            # print "'"+all_features[i+1]+"'" + ","
            kbest_features.append(all_features[i+1])

    #### uese feature important
    data = featureFormat(my_dataset, kbest_features, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.4, random_state=42)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler().fit(features_train)
    X_train_std = scaler.transform(features_train)
    X_test_std = scaler.transform(features_test)

    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=10)
    clf.fit(X_train_std, labels_train)
    pred = clf.predict(X_test_std)

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    accuscore = accuracy_score(labels_test, pred)
    precscore = precision_score(labels_test, pred)
    recascore = recall_score(labels_test, pred)
    f1score = f1_score(labels_test, pred)

    print ""+ str(k_i)+ " "+str(accuscore) + " " + str(precscore) +\
          " " + str(float(recascore)) + " "+ str(float(f1score))


    print "kbest features: ", kbest_features
    print "features importances: ", clf.feature_importances_

    # importance_features =[('poi', 1)]
    importance_features = []
    # print "feature importance: feature"
    for i in range(len(clf.feature_importances_)):
        if clf.feature_importances_[i] > 0.10:
            # print clf.feature_importances_[i], kbest_features[i+1]
            importance_features.append((kbest_features[i+1], clf.feature_importances_[i]))

    importance_features_sorted = sorted(importance_features, key=lambda x: x[1], reverse=True)
    print importance_features_sorted
    importance_features_list = zip(*importance_features_sorted)[0]
    print importance_features_list
    select_k.append((str(k_i), str(accuscore), str(precscore), str(float(recascore)), str(float(f1score)),
                     tuple(importance_features_list),
                     len(importance_features_list)
                     ))




print select_k
import pandas as pd
select_k_df = pd.DataFrame(data=select_k)
select_k_df.columns =["k", "Accuracy", "Precision", "Recall", "F1", "features with importance > 0.05", "count"]

print select_k_df
select_k_df.to_csv("p5_select_k.csv")
####################################################



############################################################
features_list =['poi',
                'to_messages',
                'expenses',
'total_payments',
'loan_advances',
'bonus',
'exercised_stock_options',
'from_ratio'
]
