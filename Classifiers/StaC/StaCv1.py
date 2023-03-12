## General imports
import pandas as pd
import numpy as np
import warnings
import time
import sys

## Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, classification_report
from sklearn.metrics import (
    accuracy_score,
    label_ranking_loss,
    coverage_error,
    average_precision_score,
    zero_one_loss,
)
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


warnings.filterwarnings("ignore")


def one_error(Y_test_oe, Y_score_oe):
    print("Function Called !")
    print(Y_test_oe.shape)
    print(Y_score_oe.shape)

    list1 = np.array(
        np.argmax(Y_score_oe, axis=1)
    )  # Find the top ranked predicted label
    print(len(list1))

    count_oe = 0

    for i in range(list1.shape[0]):  # BR uses Y_test.iloc[i,list1[i]].values
        if (
            Y_test_oe.iloc[i, list1[i]]
        ) == 0:  # If top ranked predicted label is not in the test the count it
            count_oe += 1

    return count_oe / Y_score_oe.shape[0]


class StackedChaining:
    def __init__(self, features, classes, split, label_order, name) -> None:
        self.features = features
        self.classes = classes
        self.split = split
        self.training_time = 0
        self.prediction_time = 0
        self.name = name
        self.label_order = label_order
        (
            self.X_Train_original,
            self.X_Test_original,
            self.Y_Train_All_original,
            self.Y_Test_All_original,
        ) = train_test_split(features, classes, test_size=0.3)
        self.map_idx_to_label = {}
        self.new_label_order = []
        self.num_features = len(self.features.columns)
        self.num_classes = len(self.classes.columns)
        self.metric_values = {}
        print("StackChaining Activated!")

    def generate_new_label_order(self):
        labels = self.classes.columns
        for i, label in enumerate(labels):
            self.map_idx_to_label[i + 1] = label

        for idx in self.label_order:
            self.new_label_order.append(self.map_idx_to_label[idx])

    def level0_chaining(self):
        Y_Pred_DFrame = pd.DataFrame()
        Y_Pred_Prob_DFrame = pd.DataFrame()

        pos = -1
        f_pos = self.split - 1
        if self.name == "genbase": 
            f_pos = f_pos - 1

        Y_Train_New = pd.DataFrame()
        Y_Test_New = pd.DataFrame()

        for new_lbl in self.new_label_order:
            Y_Train_New[new_lbl] = self.Y_Train_All_original[new_lbl].values
            Y_Test_New[new_lbl] = self.Y_Test_All_original[new_lbl].values

        X_Train = self.X_Train_original
        X_Test = self.X_Test_original
        print("PROPOSED STACKED CLASSIFIER CHAIN")

        print("##### LEVEL - 0 CLASSIFIER CHAIN #####")
        for each_label in self.new_label_order:
            print("LABEL in CHAIN : ", each_label)
            Y_Train = Y_Train_New[each_label]
            Y_Test = Y_Test_New[each_label]

            logreg = LogisticRegression()

            tr1_strt = time.time()
            logreg.fit(X_Train, Y_Train)
            tr1_end = time.time()

            self.training_time = self.training_time + (tr1_end - tr1_strt)

            pr1_strt = time.time()
            Y_Pred = logreg.predict(X_Test)
            pr1_end = time.time()

            self.prediction_time = self.prediction_time + (pr1_end - pr1_strt)

            Y_Pred_Prob = logreg.predict_proba(X_Test)

            pos = pos + 1
            prob_label = each_label + " PROB"
            Y_Pred_DFrame.insert(pos, each_label, Y_Pred)
            Y_Pred_Prob_DFrame.insert(pos, prob_label, Y_Pred_Prob[:, 1])

            f_pos = f_pos + 1

            X_Train.insert(f_pos, each_label, Y_Train.values)
            X_Test.insert(f_pos, each_label, Y_Pred)

        return Y_Pred_DFrame, Y_Test_New

    def level1_chaining(self, Y_Pred_DFrame):
        X_Train_L2 = self.X_Train_original
        Y_Train_All_L2 = self.Y_Train_All_original
        X_Test_L2 = self.X_Test_original

        for new_lbl in self.new_label_order:
            X_Train_L2[new_lbl] = Y_Train_All_L2[new_lbl].values
            X_Test_L2[new_lbl] = Y_Pred_DFrame[new_lbl].values

        Y_Pred_DFrame_L2 = pd.DataFrame()
        Y_Pred_Prob_DFrame_L2 = pd.DataFrame()
        pos_l2 = -1
        f_pos_l2 = self.num_classes + self.num_features
        if self.name == "yeast":
            f_pos_l2 = 114
        elif self.name == "genbase":
            f_pos_l2 = 1200
        elif self.name == "CAL500":
            f_pos_l2 = 207
        elif self.name == "Corel5k":
            f_pos_l2 = 711

        print("##### LEVEL - 1 NESTED STACKING #####")
        for each_label in self.new_label_order:
            print("LABEL in CHAIN : ", each_label)
            Y_Train_L2 = X_Train_L2[each_label]
            Y_Test_L2 = X_Test_L2[each_label]

            X_Train_L2.drop([each_label], axis=1, inplace=True)
            X_Test_L2.drop([each_label], axis=1, inplace=True)

            f_pos_l2 = f_pos_l2 - 1

            logreg = LogisticRegression()

            tr2_strt = time.time()
            logreg.fit(X_Train_L2, Y_Train_L2)
            tr2_end = time.time()

            self.training_time = self.training_time + (tr2_end - tr2_strt)

            pr2_strt = time.time()
            Y_Pred_L2 = logreg.predict(X_Test_L2)
            pr2_end = time.time()

            self.prediction_time = self.prediction_time + (pr2_end - pr2_strt)

            Y_Pred_Prob_L2 = logreg.predict_proba(X_Test_L2)

            # tr_ex_strt = time.time()
            # Y_Pred_L2_Train = logreg.predict(X_Train_L2)
            # tr_ex_end = time.time()

            # self.training_time = self.training_time + (tr_ex_end - tr_ex_strt)

            pos_l2 = pos_l2 + 1
            prob_label = each_label + " PROB"
            Y_Pred_DFrame_L2.insert(pos_l2, each_label, Y_Pred_L2)
            Y_Pred_Prob_DFrame_L2.insert(pos_l2, prob_label, Y_Pred_Prob_L2[:, 1])

            f_pos_l2 = f_pos_l2 + 1

            X_Train_L2.insert(f_pos_l2, each_label, Y_Train_L2.values)
            X_Test_L2.insert(f_pos_l2, each_label, Y_Pred_L2)

        return Y_Pred_DFrame_L2, Y_Pred_Prob_DFrame_L2

    def print_metrics(self, Y_Test_New, Y_Pred_DFrame_L2, Y_Pred_Prob_DFrame_L2):
        # print("##### OVERALL PERFORMANCE EVALUATION #####")
        # print("CLASSIFICATION REPORT : ")
        # print(classification_report(Y_Test_New, Y_Pred_DFrame_L2))
        # print(f"Y test new shape: {Y_Test_New.shape} and Y pred DFrame shape: {Y_Pred_DFrame_L2.shape}")
        h_loss_all = hamming_loss(Y_Test_New, Y_Pred_DFrame_L2)
        acc_all = 1 - h_loss_all
        self.metric_values["HAMMING LOSS"] = h_loss_all
        self.metric_values["ACCURACY"] = acc_all
        self.metric_values["SUBSET ACCURACY"] = accuracy_score(
            Y_Test_New, Y_Pred_DFrame_L2
        )
        # self.metric_values["LABEL RANKING LOSS"] = label_ranking_loss(
        #     Y_Test_New, Y_Pred_Prob_DFrame_L2
        # )
        # self.metric_values["COVERAGE ERROR"] = coverage_error(
        #     Y_Test_New, Y_Pred_Prob_DFrame_L2
        # )
        # self.metric_values["ZERO ONE LOSS"] = zero_one_loss(
        #     Y_Test_New, Y_Pred_DFrame_L2
        # )
        self.metric_values["AVERAGE PRECISION SCORE"] = average_precision_score(
             Y_Test_New, Y_Pred_Prob_DFrame_L2
        )
        # self.metric_values["LABEL RANKING APR"] = label_ranking_average_precision_score(
        #     Y_Test_New, Y_Pred_Prob_DFrame_L2
        # )
        self.metric_values["JACCARD MACRO"] = jaccard_score(
            Y_Test_New, Y_Pred_DFrame_L2, average="macro"
        )
        self.metric_values["JACCARD MICRO"] = jaccard_score(
            Y_Test_New, Y_Pred_DFrame_L2, average="micro"
        )
        self.metric_values["JACCARD SAMPLES"] = jaccard_score(
            Y_Test_New, Y_Pred_DFrame_L2, average="samples"
        )
        #one_err = one_error(Y_Test_New, Y_Pred_Prob_DFrame_L2.values)
        # self.metric_values["ONE ERROR"] = one_err
        self.metric_values["F1 SCORE MACRO"] = f1_score(
            Y_Test_New, Y_Pred_DFrame_L2, average="macro"
        )
        self.metric_values["F1 SCORE MICRO"] = f1_score(
            Y_Test_New, Y_Pred_DFrame_L2, average="micro"
        )
        self.metric_values["F1 SCORE SAMPLES"] = f1_score(
            Y_Test_New, Y_Pred_DFrame_L2, average="samples"
        )

        # print("HAMMING LOSS : ", h_loss_all)
        # print("ACCURACY : ", acc_all)
        # print("SUBSET ACCURACY : ", self.metric_values["SUBSET ACCURACY"])
        # print("LABEL RANKING LOSS : ", self.metric_values["LABEL RANKING LOSS"])
        # print("COVERAGE ERROR : ", self.metric_values["COVERAGE ERROR"])
        # print("ZERO ONE LOSS : ", self.metric_values["ZERO ONE LOSS"])
        # print("AVERAGE PRECISION SCORE : ", self.metric_values["AVERAGE PRECISION SCORE"])

        # # print("ROC AUC MICRO : ", roc_auc_score(Y_Test_New, Y_Pred_Prob_DFrame_L2, average = 'micro'))
        # # print("ROC AUC MACRO : ", roc_auc_score(Y_Test_New, Y_Pred_Prob_DFrame_L2, average = 'macro'))
        # print("LABEL RANKING APR : ", self.metric_values["LABEL RANKING APR"])
        # print("JACCARD MACRO : ", self.metric_values["JACCARD MACRO"])
        # print("JACCARD MICRO : ", self.metric_values["JACCARD MICRO"])
        # print("JACCARD SAMPLES : ", self.metric_values["JACCARD SAMPLES"])

        # print("ONE ERROR : ", one_err)
        # print("F1 SCORE MACRO : ", self.metric_values["F1 SCORE MACRO"])
        # print("F1 SCORE MICRO : ", self.metric_values["F1 SCORE MICRO"])
        # print("F1 SCORE SAMPLES : ", self.metric_values["F1 SCORE SAMPLES"])

        # print("Training Time : ", self.training_time)
        # print("Time in Predictions : ", self.prediction_time)

    def run(self):
        self.generate_new_label_order()
        Y_Pred_DFrame, Y_Test_New = self.level0_chaining()
        Y_Pred_DFrame_L2, Y_Pred_Prob_DFrame_L2 = self.level1_chaining(Y_Pred_DFrame)
        self.print_metrics(Y_Test_New, Y_Pred_DFrame_L2, Y_Pred_Prob_DFrame_L2)
