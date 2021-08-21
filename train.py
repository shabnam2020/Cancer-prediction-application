import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import f1_score

from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore") 

df = pd.read_csv("datasets/data.csv")
df.drop(["Unnamed: 32"], axis=1, inplace=True)

label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

X_train, X_test, y_train, y_test = train_test_split(df.drop(["id", "diagnosis"], axis=1),
                                                    df['diagnosis'], 
                                                    test_size=0.2, 
                                                    random_state=42)

# Logistic Regression
print("Model Name: Logistic Regression")
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_log_reg = lr_model.predict(X_test)
train_acc = round(lr_model.score(X_train, y_train) * 100, 2)
print ("Train Accuracy: " + str(train_acc) + '%')
test_acc = round(lr_model.score(X_test, y_test) * 100, 2)
print ("Test Accuracy: " + str(test_acc) + '%')
print("Test F1 Score: " + str(round(f1_score(y_test, y_pred_log_reg, average='macro'), 3)))              
print("-"*30)

# Support Vector Classifier
print("Model Name: Support Vector Classifier")
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)
train_acc = round(svc_model.score(X_train, y_train) * 100, 2)
print ("Train Accuracy: " + str(train_acc) + '%')
test_acc = round(svc_model.score(X_test, y_test) * 100, 2)
print ("Test Accuracy: " + str(test_acc) + '%')
print("Test F1 Score: " + str(round(f1_score(y_test, y_pred_svc, average='macro'), 3)))
print("-"*30)

# k-Nearest Neighbours
print("Model Name: k-Nearest Neighbours")
knn_model = KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
train_acc = round(knn_model.score(X_train, y_train) * 100, 2)
print ("Train Accuracy: " + str(train_acc) + '%')
test_acc = round(knn_model.score(X_test, y_test) * 100, 2)
print ("Test Accuracy: " + str(test_acc) + '%')
print("Test F1 Score: " + str(round(f1_score(y_test, y_pred_knn, average='macro'), 3)))
print("-"*30)

# Gaussian Naive Bayes
print("Model Name: Gaussian Naive Bayes")
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)
y_pred_gnb = gnb_model.predict(X_test)
train_acc = round(gnb_model.score(X_train, y_train) * 100, 2)
print ("Train Accuracy: " + str(train_acc) + '%')
test_acc = round(gnb_model.score(X_test, y_test) * 100, 2)
print ("Test Accuracy: " + str(test_acc) + '%')
print("Test F1 Score: " + str(round(f1_score(y_test, y_pred_gnb, average='macro'), 3)))
print("-"*30)

# Decision Tree
print("Model Name: Decision Tree")
dtree_model = DecisionTreeClassifier()
dtree_model.fit(X_train, y_train)
y_pred_decision_tree = dtree_model.predict(X_test)
train_acc = round(dtree_model.score(X_train, y_train) * 100, 2)
print ("Train Accuracy: " + str(train_acc) + '%')
test_acc = round(dtree_model.score(X_test, y_test) * 100, 2)
print ("Test Accuracy: " + str(test_acc) + '%')
print("Test F1 Score: " + str(round(f1_score(y_test, y_pred_decision_tree, average='macro'), 3)))
print("-"*30)

# Random Forest
print("Model Name: Random Forest")
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_random_forest = rf_model.predict(X_test)
train_acc = round(rf_model.score(X_train, y_train) * 100, 2)
print ("Train Accuracy: " + str(train_acc) + '%')
test_acc = round(rf_model.score(X_test, y_test) * 100, 2)
print ("Test Accuracy: " + str(test_acc) + '%')
print("Test F1 Score: " + str(round(f1_score(y_test, y_pred_random_forest, average='macro'), 3)))
print("-"*30)

# Hybrid Model:1 [kNN, DTree]
print("Model Name: Hybrid Model:1 [kNN, DTree]")
voting_model1 = VotingClassifier(estimators=[('kNN', knn_model), ('DTree', dtree_model)], voting='hard')
voting_model1.fit(X_train, y_train)
y_pred_voting1 = voting_model1.predict(X_test)
train_acc = round(voting_model1.score(X_train, y_train) * 100, 2)
print ("Train Accuracy: " + str(train_acc) + '%')
test_acc = round(voting_model1.score(X_test, y_test) * 100, 2)
print ("Test Accuracy: " + str(test_acc) + '%')
print("Test F1 Score: " + str(round(f1_score(y_test, y_pred_voting1, average='macro'), 3)))
print("-"*30)

# Hybrid Model:2 [RForest, GNB]
print("Model Name: Hybrid Model:2 [RForest, GNB]")
voting_model2 = VotingClassifier(estimators=[('RForest', rf_model), ('GNB', gnb_model)], voting='hard')
voting_model2.fit(X_train, y_train)
y_pred_voting2 = voting_model2.predict(X_test)
train_acc = round(voting_model2.score(X_train, y_train) * 100, 2)
print ("Train Accuracy: " + str(train_acc) + '%')
test_acc = round(voting_model2.score(X_test, y_test) * 100, 2)
print ("Test Accuracy: " + str(test_acc) + '%')
print("Test F1 Score: " + str(round(f1_score(y_test, y_pred_voting2, average='macro'), 3)))
print("-"*30)

# Hybrid Model:3 [LR, SVM]
print("Model Name: Hybrid Model:3 [LR, SVM]")
voting_model3 = VotingClassifier(estimators=[('LR', lr_model), ('SVM', svc_model)], voting='hard')
voting_model3.fit(X_train, y_train)
y_pred_voting3 = voting_model3.predict(X_test)
train_acc = round(voting_model3.score(X_train, y_train) * 100, 2)
print ("Train Accuracy: " + str(train_acc) + '%')
test_acc = round(voting_model3.score(X_test, y_test) * 100, 2)
print ("Test Accuracy: " + str(test_acc) + '%')
print("Test F1 Score: " + str(round(f1_score(y_test, y_pred_voting3, average='macro'), 3)))
print("#"*50, end="\n\n")

# CHOSEN MODEL
print("CHOSEN MODEL: SVM", end="\n\n")
chosen_model = voting_model2
joblib.dump(chosen_model, "./models/chosen_model.pkl")
joblib.dump(label_encoder, "./models/label_encoder.pkl")
