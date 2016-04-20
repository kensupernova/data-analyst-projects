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

##########################################################################
print "total number of data points: "+ str(len(data_dict))
###########################################################################

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

############################
### check outlier
"""
financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
 'poi', 'shared_receipt_with_poi']
"""
# checked_list = [
#                  'exercised_stock_options',
#                  "total_stock_value"
#                 ]
# checked_data = featureFormat(my_dataset, checked_list, sort_keys = True)
# x, y = targetFeatureSplit(checked_data)
# import matplotlib.pyplot as plt
# fig = plt.figure(num=None, figsize=(8, 6),dpi=100, facecolor='w', edgecolor='k')
# plot1 = fig.add_subplot(1,1,1)
# plot1.scatter(x, y)
# plt.show()

###################################################################
#### split features and labels
data = featureFormat(my_dataset, all_features, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.4, random_state=42)

##### scale the features
#minmax scale features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(features_train)
X_train_std = scaler.transform(features_train)
X_test_std = scaler.transform(features_test)

###################################################################
###### select features
### using kbest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
select = SelectKBest(chi2, k=10)
X_train_std_new = select.fit_transform(X_train_std, labels_train)
selected= select.get_support()

kbest_features= ['poi']
for i, item in enumerate(selected):
    if item:
        print "'"+all_features[i+1]+"'" + ","
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
print "features: ", kbest_features
print "features importances: ", clf.feature_importances_

importance_features =[('poi', 1)]
print "feature importance: feature"
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] >=0.00:
        print clf.feature_importances_[i], kbest_features[i+1]
        importance_features.append((kbest_features[i+1], clf.feature_importances_[i]))


importance_features_sorted = sorted(importance_features, key=lambda x: x[1])
print importance_features_sorted

features_list = ['poi', 'bonus', 'to_messages',
                 'from_ratio', 'exercised_stock_options',
                 'expenses', 'loan_advances', 'total_payments']

# # selection 1
# features_list =['poi',
#                 'to_messages',
#                 'expenses',
# 'total_payments',
# 'loan_advances',
# 'bonus',
# 'exercised_stock_options',
# 'from_ratio'
# ]

# # selection 1
# features_list =['poi',
#                 'to_messages',
#                 'expenses',
# 'total_payments',
# 'loan_advances',
# 'bonus',
# 'exercised_stock_options',
# 'from_ratio'
# ]

# selection 2
# features_list =['poi',
#                 'to_messages',
#                 'expenses',
# 'total_payments',
# 'bonus',
# 'exercised_stock_options',
# 'from_ratio'
# ]
# Accuracy: 0.83293	Precision: 0.36007	Recall: 0.32550	F1: 0.34191	F2: 0.33187
#3
# #### selection 3
# features_list = ['poi','bonus', 'shared_receipt_with_poi',
#                  'total_stock_value', 'from_ratio', 'salary',
#                  'exercised_stock_options', 'restricted_stock']
#
# Accuracy: 0.81221	Precision: 0.30429	Recall: 0.24450	F1: 0.27114	F2: 0.25450
#### selection 4

# features_list =['poi','bonus', 'other', 'from_ratio', 'total_payments',
#                 'total_stock_value', 'restricted_stock', 'salary']
# Accuracy: 0.81221	Precision: 0.30429	Recall: 0.24450	F1: 0.27114	F2: 0.25450

# #### selection 5
#
# features_list = ['poi', 'bonus', 'other', 'from_ratio', 'exercised_stock_options',
#                  'salary', 'total_stock_value', 'total_payments']



print features_list
######################################## reselect the data
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.4, random_state=42)
print "length of training sample: ", len(features_train)
print "length of test sample: ", len(features_test)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(features_train)
X_train_std = scaler.transform(features_train)
X_test_std = scaler.transform(features_test)

############
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()
from sklearn.tree import DecisionTreeClassifier
clf2 = DecisionTreeClassifier(random_state=1)
from sklearn.cluster import KMeans
clf3 = KMeans(n_clusters=2)
from sklearn.neighbors import KNeighborsClassifier
clf4 = KNeighborsClassifier()
from sklearn.ensemble import AdaBoostClassifier
clf5= AdaBoostClassifier(random_state=12)
from sklearn.ensemble import RandomForestClassifier
clf6 = RandomForestClassifier(random_state=12)
from sklearn.svm import SVC
clf7 = SVC(C=10, kernel='rbf')

estimators = [clf1, clf2, clf3, clf4, clf5, clf6, clf7]
names = [
        "Gaussian NB",
         "Decision Tree Classifier",
         "Kmeans",
         "KNeighbors Classifier",
         "AdaBoost Classifer",
         "Random Forest Classifier",
         "SVC"
         ]

gaussnb = []
decissiontree = []
kmeans = []
kneighbor = []
adaboost = []
randomforest = []
svc = []

colors = ["black", "k", "g", "r", "c", "y","blue"]



# evaluate the algorithm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


################### get the metrics
scores = [accuracy_score, precision_score, recall_score, f1_score]
def get_metrics(X_train, X_test):
    gaussnb = []
    decissiontree = []
    kmeans = []
    kneighbor = []
    adaboost = []
    randomforest = []
    svc = []
    lists = [gaussnb, decissiontree, kmeans, kneighbor, adaboost, randomforest, svc]
    for i, model in enumerate(estimators):
        for score in scores:
            estimator = model
            estimator.fit(X_train, labels_train)
            pred = estimator.predict(X_test)
            lists[i].append(score(labels_test, pred))
    import pandas as pd
    scores_df = pd.DataFrame(data = lists)
    scores_df.index = ["gaussnb", "decision tree", "kmeans",
                       "kneighbors", "adaboost", "random forest", "svc"]
    scores_df.columns=["Accuracy", "Precision", "Recall", "F1"]
    return scores_df

scores_df = get_metrics(X_train_std, X_test_std)
print scores_df
print "the selected clf: "
print clf5
scores_df.to_csv("p5_scores df.csv")

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!

"""
selected_clf = RandomForestClassifier()
s_n_estimators = [5, 10, 15, 20, 30, 50, 75, 100]
s_max_features = ["auto", 'sqrt', "log2"]
s_min_samples_split = [2, 3, 4, 5, 6, 7, 8,]

s_min_samples_leaf = [1, 2, 3, 4]
s_max_depth = [100]
choices = [s_n_estimators, s_max_features, s_min_samples_split]
import itertools
from time import time
combinations = list(itertools.product(*choices))
# print combinations

combinations_scores= {}
for item in combinations:
    score_list=[]
    t0 = time()
    for score in scores:
        selected_clf = RandomForestClassifier(n_estimators= item[0],
                                              max_features= item[1], min_samples_split=item[2])

        selected_clf.fit(X_train_std, labels_train)
        pred = selected_clf.predict(X_test_std)
        score_list.append(score(labels_test, pred))
    # print "consumed time: ", round(time() -t0, 3), " s"
    ### add the consumed time to the first element of the list
    score_list.append(round(time() -t0, 3))
    combinations_scores[item] = score_list

"""
selected_clf = AdaBoostClassifier()
s_algorithm=["SAMME", "SAMME.R"]
s_n_estimators = [10, 50, 100, 150, 200]
s_learning_rate = [0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4,  5]

s_min_samples_leaf = [1, 2, 3, 4]
s_max_depth = [100]
choices = [s_algorithm, s_n_estimators, s_learning_rate]
import itertools
from time import time
combinations = list(itertools.product(*choices))
# print combinations

combinations_scores= {}
for item in combinations:
    score_list=[]
    t0 = time()
    for score in scores:
        selected_clf = AdaBoostClassifier(algorithm= item[0],
           n_estimators= item[1], learning_rate=item[2], random_state=12)

        selected_clf.fit(X_train_std, labels_train)
        pred = selected_clf.predict(X_test_std)
        score_list.append(score(labels_test, pred))
    # print "consumed time: ", round(time() -t0, 3), " s"
    ### add the consumed time to the first element of the list
    score_list.append(round(time() -t0, 3))
    combinations_scores[item] = score_list


print combinations_scores
#### choose the best combination
good_combinations = {}
max_accuray = max(combinations_scores.values(), key = lambda item:item[0])[0]
max_precision = max(combinations_scores.values(), key = lambda item:item[1])[1]
max_recall = max(combinations_scores.values(), key = lambda item:item[2])[2]
max_f1 =  max(combinations_scores.values(), key = lambda item:item[3])[3]
print 'maximum precision value: ', max_precision
print "maximum recall value: ", max_recall
for key, value in combinations_scores.iteritems():
    if value[1] >=0.35 and value[2] >= 0.35:
        good_combinations[key] = value

print good_combinations
sorted(good_combinations.values(), key=lambda x:x[2])
print good_combinations.items()
selected_combination = good_combinations.items()[len(good_combinations)-1]
print "selected parameters: "
print selected_combination
selected_p = selected_combination[0]
print selected_p
item = selected_p

# selected_clf = RandomForestClassifier(n_estimators= item[0],
#                                               max_features= item[1], random_state=12)

selected_clf = AdaBoostClassifier(algorithm= item[0],
           n_estimators= item[1], learning_rate=item[2], random_state=12)

#### validation
from sklearn import cross_validation
cross_validation_scores = cross_validation.cross_val_score(selected_clf, features, labels, cv=5)
print "cross validation scores: " + str(cross_validation_scores)
cross_mean = round(cross_validation_scores.mean(), 2)
cross_std = round(cross_validation_scores.std()*2, 4)

print "mean and 95% confidence interval: "
print "%s +/- %s" % (str(cross_mean), str(cross_std))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
######## decide the algorithm

estimators = [("scaler", scaler),
    ("RandomForest" ,selected_clf)]
from sklearn.pipeline import Pipeline
clf = Pipeline(estimators)

print clf
dump_classifier_and_data(clf, my_dataset, features_list)