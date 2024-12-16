# Support libs
import copy
import os
import random
import time

# Name of the folder in which look for tabular (CSV) datasets

# Works only with anomaly detection (no multi-class)
# ------- GLOBAL VARS -----------
import numpy as numpy
import pandas
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting
from logitboost import LogitBoost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from cinnabar.calibration.CalibrationStrategy import PlattScaling, IsotonicScaling, BBQScaling, HistogramScaling
from cinnabar.prediction_rejection.PredictionRejection import SufficientlySafe, ValueAware, compute_value, \
    EntropyRejection, EnsembleRejection, SPROUTStrategy, SPROUTRejection
from cinnabar.utils.dataset_utils import read_tabular_dataset, read_binary_tabular_dataset
from cinnabar.utils.general_utils import get_classifier_name, current_ms, compute_rejection_metrics, compute_clf_metrics

# True if you want to force binary classification
FORCE_BINARY = True
# Cost of rejections
REJECT_COST = 0

# Set random seed for reproducibility
random.seed(42)
numpy.random.seed(42)


def get_cost_matrixes() -> list:
    """
    Returns the list of cost matrixes to be used in the experiments
    :return:
    """
    cost_matrixes = []
    if FORCE_BINARY:
        # items are list of 4 items: [TP, FP, FN, TN]
        cost_matrixes.append([100, -1, -10000, 1])
    else:
        # items are list of 2 items: [correct class, misclassification]
        cost_matrixes.append([1, -100])
    return cost_matrixes


def get_rejection_strategies(cost_matrix) -> list:
    """
    returns the list of prediction rejection strategies to be used in experiments
    :param cost_matrix: the cost matrix to be used (if value-aware)
    :return: a list of objects
    """
    value_thresholds = numpy.arange(0, 1, 0.01, dtype=float)
    rej_list = [
        EntropyRejection(cost_matrix=cost_matrix, reject_cost=REJECT_COST),
        SufficientlySafe(cost_matrix=cost_matrix, reject_cost=REJECT_COST, max_iterations=50),
        ValueAware(cost_matrix=cost_matrix, reject_cost=REJECT_COST,
                   candidate_thresholds=value_thresholds, normal_tag=0)]
    return rej_list


# ----------------------- MAIN ROUTINE ---------------------
# This script replicates experiments done for testing the robustness of confidence ensembles
if __name__ == '__main__':

    # Iterating over datasets
    for dataset_file in os.listdir("sample_data"):
        # if file is a CSV, it is assumed to be a dataset to be processed
        if dataset_file.endswith(".csv"):
            dataset_name = dataset_file.replace(".csv", "")
            # Read dataset
            if FORCE_BINARY:
                data_dict = read_binary_tabular_dataset(dataset_name=os.path.join("sample_data", dataset_file),
                                                        label_name="multilabel",
                                                        train_size=0.5, val_size=0.2,
                                                        shuffle=True, l_encoding=True, normal_tag="normal")
            else:
                data_dict = read_tabular_dataset(dataset_name=os.path.join("sample_data", dataset_file),
                                                 label_name="multilabel",
                                                 train_size=0.5, val_size=0.2,
                                                 shuffle=True, l_encoding=True)

            # Loop for training and testing each classifier
            contamination = 1 - data_dict["normal_perc"] if FORCE_BINARY else None
            for uncal_clf in [DecisionTreeClassifier()]:
                clf_name = get_classifier_name(uncal_clf)
                # Training the algorithm to get a model
                start_time = current_ms()
                uncal_clf.fit(data_dict["x_train"], data_dict["y_train"])
                train_time = current_ms() - start_time

                # Loop over calibration strategies (None means no calibration)
                for calibrated_classifier in [PlattScaling(uncal_clf)]:
                    cs_name = calibrated_classifier.get_name()
                    classifier = calibrated_classifier
                    classifier.fit(data_dict["x_train"], data_dict["y_train"])
                    test_pred = classifier.predict(data_dict["x_test"])
                    clf_metrics = compute_clf_metrics(y_true=data_dict["y_test"], y_clf=test_pred)

                    # Loop over cost matrixes
                    cost_matrix_list = get_cost_matrixes()
                    for cost_matrix in cost_matrix_list:
                        cost_mat_name = ";".join([str(x) for x in cost_matrix])
                        clf_value = compute_value(data_dict["y_test"], test_pred,
                                                  cost_matrix, REJECT_COST, None, 0)
                        print("'%s - %s': accuracy %.4f, value %.3f" %
                              (clf_name, cs_name, clf_metrics['acc'], clf_value))

                        # Creating Tuned Classifier
                        tuned_classifier = TunedThresholdClassifierCV(
                            estimator=classifier,
                            scoring=make_scorer(compute_value, cost_matrix=cost_matrix, reject_value=REJECT_COST,
                                                reject_tag=None, normal_tag=0),
                            store_cv_results=True,
                        )
                        tuned_classifier.fit(data_dict["x_train"], data_dict["y_train"])
                        tune_desc = tuned_classifier.best_threshold_
                        tuned_val_proba = classifier.predict_proba(data_dict["x_val"])
                        tuned_val_pred = classifier.predict(data_dict["x_val"])
                        tuned_test_proba = classifier.predict_proba(data_dict["x_test"])
                        tuned_test_pred = tuned_classifier.predict(data_dict["x_test"])
                        tuned_clf_metrics = compute_clf_metrics(y_true=data_dict["y_test"], y_clf=tuned_test_pred)
                        tuned_clf_value = compute_value(data_dict["y_test"], tuned_test_pred,
                                                        cost_matrix, REJECT_COST, None, 0)
                        print("\tTuned: accuracy %.4f, value %.3f" % (tuned_clf_metrics['acc'], tuned_clf_value))

                        # Loop over rejection strategies
                        rej_strategy_list = get_rejection_strategies(cost_matrix)
                        for rej_strategy in rej_strategy_list:
                            reject_name = rej_strategy.get_name()
                            reject_desc = rej_strategy.describe()


                            # Otherwise we can move ahead
                            start_ms = current_ms()
                            rej_strategy.fit(proba=tuned_val_proba, y_pred=tuned_val_pred,
                                             y_true=data_dict["y_val"],
                                             verbose=False)
                            rej_pred_y = rej_strategy.apply(test_proba=tuned_test_proba, x_test=data_dict["x_test"],
                                                            test_label=test_pred)
                            rej_time = current_ms() - start_ms
                            value = compute_value(data_dict["y_test"], rej_pred_y,
                                                  cost_matrix, REJECT_COST, None, 0)
                            rej_clf_metrics = compute_clf_metrics(y_true=data_dict["y_test"],
                                                                  y_clf=rej_pred_y)
                            rej_clf_metrics['time'] = rej_time
                            rej_metrics = compute_rejection_metrics(y_true=data_dict["y_test"],
                                                                    y_wrapper=rej_pred_y,
                                                                    y_clf=test_pred)

                            print(
                                "\twith '%s': \tvalue %.3f, accuracy %.4f, misc %.4f, rejections %.4f, corr. rej. %.3f, misc gain %.3f" %
                                (reject_name, value, rej_metrics['alpha_w'], rej_metrics['eps_w'],
                                 rej_metrics['phi'], rej_metrics['phi_m_ratio'], rej_metrics['eps_gain']))

                classifier = None
