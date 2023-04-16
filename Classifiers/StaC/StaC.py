import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.metrics import hamming_loss, classification_report
# from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import accuracy_score, label_ranking_loss, coverage_error, average_precision_score, zero_one_loss
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import jaccard_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time

warnings.filterwarnings('ignore')

# dset = pd.read_csv("full_emotions.csv")
# dset = pd.read_csv("full_scenes.csv")
# dset = pd.read_csv("full_yeast.csv")
# dset = pd.read_csv("full_birds.csv")
# dset = pd.read_csv("full_genbase.csv")
# dset = pd.read_csv("full_corel5k.csv")
# dset = pd.read_csv("full_cal500.csv")
# dset = pd.read_csv("full_rcv1.csv")
# dset = pd.read_csv("full_flags.csv")
dset = pd.read_csv("../../Datasets/mediamill_120.csv")
print(dset.shape)
print(dset.head(5))

# dset_features = dset.iloc[:, 1 : 73] # EMOTIONS
# dset_features = dset.iloc[:, 1 : 295] # SCENES
# dset_features = dset.iloc[:, 1 : 104] # YEAST
# dset_features = dset.iloc[:, 1 : 261] # BIRDS
# dset_features = dset.iloc[:, 2 : 1187] # GENBASE
# dset_features = dset.iloc[:, 1 : 500] # COREL 5 K
# dset_features = dset.iloc[:, 1 : 69] # CAL 500
# dset_features = dset.iloc[:, 0 : 944] # RCV 1 S 1
# dset_features = dset.iloc[:, 1 : 20] # FLAGS
dset_features = dset.iloc[:, 1 : 121] # MEDIA MILL
features = dset_features.columns
# dset_classes = dset.iloc[:, 73 : ] # EMOTIONS
# dset_classes = dset.iloc[:, 295 : ] # SCENES
# dset_classes = dset.iloc[:, 104 : ] # YEAST
# dset_classes = dset.iloc[:, 261 : ] # BIRDS
# dset_classes = dset.iloc[:, 1187 : ] # GENBASE
# dset_classes = dset.iloc[:, 500 : ] # COREL 5 K
# dset_classes = dset.iloc[:, 69 : ] # CAL 500
# dset_classes = dset.iloc[:, 944 : ] # RCV 1 S 1
# dset_classes = dset.iloc[:, 20 : ] # FLAGS
dset_classes = dset.iloc[:, 121 : ] # MEDIA MILL
classes = dset_classes.columns

print(features)
print(classes)

X_Train, X_Test, Y_Train_All, Y_Test_All = train_test_split(dset_features, dset_classes, test_size = 0.3)
print(X_Train.shape)
print(X_Train.head(5))
print(Y_Train_All.shape)
print(Y_Train_All.head(5))
print(X_Test.shape)
print(X_Test.head(5))
print(Y_Test_All.shape)
print(Y_Test_All.head(5))

X_Train_original = X_Train
X_Test_original = X_Test
Y_Train_All_original = Y_Train_All
Y_Test_All_original = Y_Test_All

Y_Pred_DFrame = pd.DataFrame()
Y_Pred_Prob_DFrame = pd.DataFrame()
pos = -1
t_tr = 0
t_pr = 0

# f_pos = 71 # EMOTIONS
# f_pos = 293 # SCENES
# f_pos = 102 # YEAST
# f_pos = 259 # BIRDS
# f_pos = 1184 # GENBASE 
# f_pos = 498 # COREL 5 K
# f_pos = 67 # CAL 500
# f_pos = 943 # RCV 1 S 1
# f_pos = 18 # FLAGS
f_pos = 119 # MEDIA MILL

print("PROPOSED STACKED CLASSIFIER CHAIN")

print("##### LEVEL - 0 CLASSIFIER CHAIN #####")
for each_label in classes:
    print("LABEL in CHAIN : ", each_label)
    Y_Train = Y_Train_All[each_label]
    Y_Test = Y_Test_All[each_label]
      
    logreg = LogisticRegression()
    
    tr1_strt = time.time()
    logreg.fit(X_Train, Y_Train)
    tr1_end = time.time()
    
    t_tr = t_tr + (tr1_end - tr1_strt)
    
    pr1_strt = time.time()
    Y_Pred = logreg.predict(X_Test)
    pr1_end = time.time()
    
    t_pr = t_pr + (pr1_end - pr1_strt)
    
    Y_Pred_Prob = logreg.predict_proba(X_Test)
    
    pos = pos + 1
    prob_label = each_label + " PROB"
    Y_Pred_DFrame.insert(pos, each_label, Y_Pred)
    Y_Pred_Prob_DFrame.insert(pos, prob_label, Y_Pred_Prob[:, 1])
    
    f_pos = f_pos + 1
    tr_new_col_vals = Y_Train.values
    
    X_Train.insert(f_pos, each_label, tr_new_col_vals)
    X_Test.insert(f_pos, each_label, Y_Pred)

print(Y_Pred_DFrame.head(5))
print(Y_Pred_DFrame.info())
print(Y_Pred_Prob_DFrame.head(5))
print(Y_Pred_Prob_DFrame.info())

X_Train_L2 = X_Train_original
Y_Train_All_L2 = Y_Train_All_original
X_Test_L2 = X_Test_original
Y_Test_All_L2 = Y_Test_All_original

for each_label in classes:
    X_Train_L2[each_label] = Y_Train_All_L2[each_label].values
    X_Test_L2[each_label] = Y_Pred_DFrame[each_label].values

Y_Pred_DFrame_L2 = pd.DataFrame()
Y_Pred_Prob_DFrame_L2 = pd.DataFrame()
pos_l2 = -1

# f_pos_l2 = 77 # EMOTIONS
# f_pos_l2 = 299 # SCENES
# f_pos_l2 = 114 # YEAST
# f_pos_l2 = 278 # BIRDS
# f_pos_l2 = 1200 # GENBASE
# f_pos_l2 = 711 # COREL 5 K
# f_pos_l2 = 207 # CAL 500
# f_pos_l2 = 1025 # RCV 1 S 1
# f_pos_l2 = 25 # FLAGS
f_pos_l2 = 219 # MEDIA MILL

print("##### LEVEL - 1 CLASSIFIER CHAIN #####")


for each_label in classes:
    print("LABEL in CHAIN : ", each_label)
    Y_Train_L2 = X_Train_L2[each_label]
    Y_Test_L2 = X_Test_L2[each_label]
    
    X_Train_L2.drop([each_label], axis = 1, inplace = True)
    X_Test_L2.drop([each_label], axis = 1, inplace = True)
    
    f_pos_l2 = f_pos_l2 - 1
    
    logreg = LogisticRegression()
    
    tr2_strt = time.time()
    logreg.fit(X_Train_L2, Y_Train_L2)
    tr2_end = time.time()
    
    t_tr = t_tr + (tr2_end - tr2_strt)
    
    pr2_strt = time.time()
    Y_Pred_L2 = logreg.predict(X_Test_L2)
    pr2_end = time.time()
    
    t_pr = t_pr + (pr2_end - pr2_strt)
    
    Y_Pred_Prob_L2 = logreg.predict_proba(X_Test_L2)
    
    pos_l2 = pos_l2 + 1
    prob_label = each_label + " PROB"
    Y_Pred_DFrame_L2.insert(pos_l2, each_label, Y_Pred_L2)
    Y_Pred_Prob_DFrame_L2.insert(pos_l2, prob_label, Y_Pred_Prob_L2[:, 1])
    
    f_pos_l2 = f_pos_l2 + 1
    print(f_pos_l2)
    tr_new_col_vals = Y_Train_L2.values
    
    X_Train_L2.insert(f_pos_l2, each_label, tr_new_col_vals)
    X_Test_L2.insert(f_pos_l2, each_label, Y_Pred_L2)

print(Y_Pred_DFrame_L2.head(5))
print(Y_Pred_DFrame_L2.info())
print(Y_Pred_Prob_DFrame_L2.head(5))
print(Y_Pred_Prob_DFrame_L2.info())

print("##### OVERALL PERFORMANCE EVALUATION #####")
print("CLASSIFICATION REPORT : ")
print(classification_report(Y_Test_All, Y_Pred_DFrame_L2))

h_loss_all = hamming_loss(Y_Test_All, Y_Pred_DFrame_L2)
acc_all = 1 - h_loss_all
print("HAMMING LOSS : ", h_loss_all)
print("ACCURACY : ", acc_all)

print("SUBSET ACCURACY : ", accuracy_score(Y_Test_All, Y_Pred_DFrame_L2))
print("LABEL RANKING LOSS : ", label_ranking_loss(Y_Test_All, Y_Pred_Prob_DFrame_L2))
print("COVERAGE ERROR : ", coverage_error(Y_Test_All, Y_Pred_Prob_DFrame_L2))
print("ZERO ONE LOSS : ", zero_one_loss(Y_Test_All, Y_Pred_DFrame_L2))
print("AVERAGE PRECISION SCORE : ", average_precision_score(Y_Test_All, Y_Pred_Prob_DFrame_L2))

# print("ROC AUC MICRO : ", roc_auc_score(Y_Test_All, Y_Pred_Prob_DFrame_L2, average = 'micro'))
# print("ROC AUC MACRO : ", roc_auc_score(Y_Test_All, Y_Pred_Prob_DFrame_L2, average = 'macro'))

print("LABEL RANKING APR : ", label_ranking_average_precision_score(Y_Test_All, Y_Pred_Prob_DFrame_L2))
print("JACCARD MACRO : ", jaccard_score(Y_Test_All, Y_Pred_DFrame_L2, average = 'macro'))
print("JACCARD MICRO : ", jaccard_score(Y_Test_All, Y_Pred_DFrame_L2, average = 'micro'))
print("JACCARD SAMPLES : ", jaccard_score(Y_Test_All, Y_Pred_DFrame_L2, average = 'samples'))

def one_error(Y_test_oe,Y_score_oe):
    print("Function Called !")
    print(Y_test_oe.shape)
    print(Y_score_oe.shape)
    
    list1=np.array(np.argmax(Y_score_oe,axis=1))#Find the top ranked predicted label
    print(len(list1))
    
    count_oe=0
    
    for i in range(list1.shape[0]): # BR uses Y_test.iloc[i,list1[i]].values
        if(Y_test_oe.iloc[i,list1[i]])==0:#If top ranked predicted label is not in the test the count it
            count_oe+=1
    
    return count_oe/Y_score_oe.shape[0]

one_err = one_error(Y_Test_All, Y_Pred_Prob_DFrame_L2.values)
print("ONE ERROR : ", one_err)

print("F1 MACRO : ", f1_score(Y_Test_All, Y_Pred_DFrame_L2, average = 'macro'))
print("F1 MICRO : ", f1_score(Y_Test_All, Y_Pred_DFrame_L2, average = 'micro'))
print("F1 SAMPLES : ", f1_score(Y_Test_All, Y_Pred_DFrame_L2, average = 'samples'))

print("Training Time : ", t_tr)
print("Time in Predictions : ", t_pr)