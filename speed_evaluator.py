#%% IMPORTS
from mlxtend.plotting import plot_decision_regions
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% LOAD DATA

LEN_FEATURE = 1000
def get_feature(LEN_FEATURE):
    grade = np.random.uniform(low = 0, high = 1, size = LEN_FEATURE)
    bumpiness = np.random.uniform(low = 0, high = 1, size = LEN_FEATURE)
    def _get_feature(i):
        feature = [grade[i], bumpiness[i]]
        return np.asarray(feature)
    
    list_feature = [_get_feature(i) for i in range(LEN_FEATURE)]
    arr_feature = np.asarray(list_feature)
    return arr_feature

def get_label(arr_feature):
    def _eval_feature(feature):
        if feature[0]**4  + feature[1]**3 > 0.5: # grade^4 + bumpiness^3 > 0.5 => drive slow
            return 1
        return 0 # drive fast otherwise
    list_label = [_eval_feature(feature) for feature in arr_feature]
    arr_label = np.asarray(list_label)
    return arr_label

feature_train, feature_test = get_feature(LEN_FEATURE), get_feature(LEN_FEATURE)
label_train, label_test = get_label(feature_train), get_label(feature_test)

#%% NAIVE BAYES

# Fitting the Classifier
clf = GaussianNB()
clf.fit(feature_train, label_train)

# Making a Prediction
pred = clf.predict([[0.25, 0.25]]) # correct ans == fast, since 0.25^2 < 0.25
print(pred)

# Evaluating the Classifier
score = clf.score(feature_test, label_test)
print('Score for the Classifier= {}'.format(score))

#%% SVM LINEAR
clf = svm.SVC(kernel = 'linear')
clf.fit(feature_train, label_train)
ax = plot_decision_regions(feature_train, label_train, clf = clf, legend = 1)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Slow','Fast'])
plt.xlabel('Grade')
plt.ylabel('Bumpiness')
plt.title('NB on Speed Evaluation')

plt.show()
# Prediction
pred = clf.predict([[0.25, 0.25]])
print(pred)

# Evaluating the Classifier
score = clf.score(feature_test, label_test)
print('Score for the Classifier= {}'.format(score))