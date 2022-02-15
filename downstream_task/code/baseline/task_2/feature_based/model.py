from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import GridSearchCV
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default = "LR",
    type = str
)
args = parser.parse_args()


def dice(set_1, set_2):
    return 2 * len(set_1 & set_2) / (len(set_1) + len(set_2))


patients = json.load(open("../../../../../PMC-Patients_collection/meta_data/PMC-Patients.json", "r"))
uid2age = {}
uid2gender = {}
for patient in patients:
    uid = patient['patient_uid']
    uid2gender[uid] = patient['gender']
    age = 0
    for (value, unit) in patient['age']:
        if unit == "year":
            age += value
        if unit == "month":
            age += value / 12
        if unit == "week":
            age += value / 52
        if unit == "day":
            age += value / 365
        if unit == "hour":
            age += value / 365 / 24
    uid2age[uid] = age

uid2NER = json.load(open("NER.json", "r"))

train_ins = json.load(open("../../../../datasets/task_2_patient2patient_similarity/PPS_train.json", "r"))
X_train = []
Y_train = []
for ins in train_ins:
    uid_1, uid_2, label = ins
    #label = 1 if label == 2 else label
    age = abs(uid2age[uid_1] - uid2age[uid_2])
    gender = int(uid2gender[uid_1] == uid2gender[uid_2])
    NER_sim = dice(set(uid2NER[uid_1]), set(uid2NER[uid_2]))
    #X_train.append([age, gender, NER_sim])
    X_train.append([NER_sim])
    Y_train.append(label)

dev_ins = json.load(open("../../../../datasets/task_2_patient2patient_similarity/PPS_dev.json", "r"))
X_dev = []
Y_dev = []
for ins in dev_ins:
    uid_1, uid_2, label = ins
    #label = 1 if label == 2 else label
    age = abs(uid2age[uid_1] - uid2age[uid_2])
    gender = int(uid2gender[uid_1] == uid2gender[uid_2])
    NER_sim = dice(set(uid2NER[uid_1]), set(uid2NER[uid_2]))
    #X_dev.append([age, gender, NER_sim])
    X_dev.append([NER_sim])
    Y_dev.append(label)

test_ins = json.load(open("../../../../datasets/task_2_patient2patient_similarity/PPS_test.json", "r"))
X_test = []
Y_test = []
for ins in test_ins:
    uid_1, uid_2, label = ins
    #label = 1 if label == 2 else label
    age = abs(uid2age[uid_1] - uid2age[uid_2])
    gender = int(uid2gender[uid_1] == uid2gender[uid_2])
    NER_sim = dice(set(uid2NER[uid_1]), set(uid2NER[uid_2]))
    #X_test.append([age, gender, NER_sim])
    X_test.append([NER_sim])
    Y_test.append(label)


if args.model == "SVM":
    model = svm.SVC(random_state = 21)
    parameters = {"gamma": [1e-4, 1e-3, 1e-2, .1, 1], 'C': [.01, .1, 1, 10, 100]}
if args.model == "LR":
    model = LogisticRegression(random_state = 21)
    parameters = {"penalty": ['l1', 'l2'], 'C': [.01, .1, 1, 10, 100], "multi_class": ['ovr', 'multinomial'], \
        "solver": ["liblinear", "lbfgs", "saga"]}

train_index = [list(range(len(X_train)))]
dev_index = [list(range(len(X_train), len(X_train) + len(X_dev)))]
cv = zip(train_index, dev_index)
X = X_train + X_dev
Y = Y_train + Y_dev
clf = GridSearchCV(model, parameters, cv = cv, return_train_score = True, n_jobs = 20, error_score = np.nan)

clf.fit(X, Y)
print(clf.best_params_)
print(clf.best_score_)

test_acc = clf.score(X_test, Y_test)
print(test_acc)
