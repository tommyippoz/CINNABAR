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
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from cinnabar.calibration.CalibrationStrategy import PlattScaling, IsotonicScaling, BBQScaling, HistogramScaling
from cinnabar.prediction_rejection.PredictionRejection import SufficientlySafe, ValueAware, compute_value, \
    EntropyRejection, EnsembleRejection
from cinnabar.utils.dataset_utils import read_tabular_dataset, read_binary_tabular_dataset
from cinnabar.utils.general_utils import get_classifier_name, current_ms, compute_rejection_metrics, compute_clf_metrics

CSV_FOLDER = "input_folder"
# Name of the column that contains the label in the tabular (CSV) dataset
LABEL_NAME = 'multilabel'
# Name of the 'normal' class in datasets. This will be used only for binary classification (anomaly detection)
NORMAL_TAG = 'normal'
# Name of the file in which outputs of the analysis will be saved
SCORES_FILE = "cinnabar_results.csv"
# Percentage of test data wrt train data
TVT_SPLIT = [0.5, 0.2, 0.3]
# True if debug information needs to be shown
VERBOSE = True
# True if you want to force binary classification
FORCE_BINARY = True
# Cost of rejections
REJECT_COST = -1

# Set random seed for reproducibility
random.seed(42)
numpy.random.seed(42)


# --------- SUPPORT FUNCTIONS ---------------
def get_learners(cont_perc) -> list:
    """
    Function to get a learner to use, given its string tag
    :param cont_perc: percentage of anomalies in the training set, required for unsupervised classifiers from PYOD
    :return: the list of classifiers to be trained
    """
    base_learners = [
        XGBClassifier(n_estimators=100),
        LinearDiscriminantAnalysis(),
        DecisionTreeClassifier(),
        Pipeline([("norm", MinMaxScaler()), ("gnb", GaussianNB())]),
        RandomForestClassifier(n_estimators=100),
        LogisticRegression(),
        ExtraTreesClassifier(n_estimators=100),
        ConfidenceBoosting(clf=DecisionTreeClassifier()),
    ]

    return base_learners


def get_calibrators(classifier, labels) -> list:
    """
    Returns the functions used for post-hoc calibration. If none, just return [None]
    :return:
    """
    return [PlattScaling(classifier),
            IsotonicScaling(classifier),
            HistogramScaling(classifier, n_bins=10, threshold=0.5, labels=labels),
            BBQScaling(classifier, threshold=0.5, labels=labels),
            None]


def get_cost_matrixes() -> list:
    """
    Returns the list of cost matrixes to be used in the experiments
    :return:
    """
    cost_matrixes = []
    if FORCE_BINARY:
        # items are list of 4 items: [TP, FP, FN, TN]
        cost_matrixes.append([10, -5, -1000, 1])
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
    value_thresholds = numpy.arange(0.3, 1, 0.01, dtype=float)
    rej_list = [EntropyRejection(cost_matrix=cost_matrix, reject_cost=REJECT_COST),
                SufficientlySafe(cost_matrix=cost_matrix, reject_cost=REJECT_COST,
                                 max_iterations=50),
                ValueAware(cost_matrix=cost_matrix, reject_cost=REJECT_COST,
                           candidate_thresholds=value_thresholds, normal_tag=NORMAL_TAG)]
    # Ensemble Rejectors
    rejectors = [
        EnsembleRejection(cost_matrix=cost_matrix, reject_cost=REJECT_COST, rejectors=rej_list, strategy='one'),
        EnsembleRejection(cost_matrix=cost_matrix, reject_cost=REJECT_COST, rejectors=rej_list, strategy='two'),
        EnsembleRejection(cost_matrix=cost_matrix, reject_cost=REJECT_COST, rejectors=rej_list, strategy='all'),
        ]
    # Adds base rejectors
    for rej in rej_list:
        rejectors.append(rej)
    return rejectors


# ----------------------- MAIN ROUTINE ---------------------
# This script replicates experiments done for testing the robustness of confidence ensembles
if __name__ == '__main__':

    # This is for checkpointing experiments, otherwise it starts every time from scratch
    existing_exps = None
    if os.path.exists(SCORES_FILE):
        existing_exps = pandas.read_csv(SCORES_FILE, usecols=['dataset_tag', 'calibration',
                                                              'reject_strategy', 'cost_matrix', 'clf_name'])

    # Iterating over datasets
    for dataset_file in os.listdir(CSV_FOLDER):
        # if file is a CSV, it is assumed to be a dataset to be processed
        if dataset_file.endswith(".csv"):
            dataset_name = dataset_file.replace(".csv", "")
            # Read dataset
            if FORCE_BINARY:
                data_dict = read_binary_tabular_dataset(dataset_name=os.path.join(CSV_FOLDER, dataset_file),
                                                        label_name=LABEL_NAME,
                                                        train_size=TVT_SPLIT[0], val_size=TVT_SPLIT[1],
                                                        shuffle=True, l_encoding=True, normal_tag=NORMAL_TAG)
            else:
                data_dict = read_tabular_dataset(dataset_name=os.path.join(CSV_FOLDER, dataset_file),
                                                 label_name=LABEL_NAME,
                                                 train_size=TVT_SPLIT[0], val_size=TVT_SPLIT[1],
                                                 shuffle=True, l_encoding=True)

            # Loop for training and testing each classifier
            contamination = 1 - data_dict["normal_perc"] if FORCE_BINARY else None
            learners = get_learners(contamination)
            exp_i = 1
            for uncal_clf in learners:
                clf_name = get_classifier_name(uncal_clf)
                # Training the algorithm to get a model
                start_time = current_ms()
                uncal_clf.fit(data_dict["x_train"], data_dict["y_train"])
                train_time = current_ms() - start_time

                # Loop over calibration strategies (None means no calibration)
                calibration_list = get_calibrators(uncal_clf, data_dict["label_names"])
                for calibrated_classifier in calibration_list:
                    if calibrated_classifier is None:
                        cs_name = "none"
                        classifier = copy.deepcopy(uncal_clf)
                    else:
                        cs_name = calibrated_classifier.get_name()
                        classifier = calibrated_classifier
                        classifier.fit(data_dict["x_train"], data_dict["y_train"])
                    val_proba = classifier.predict_proba(data_dict["x_val"])
                    val_pred = classifier.predict(data_dict["x_val"])
                    test_proba = classifier.predict_proba(data_dict["x_test"])
                    test_pred = classifier.predict(data_dict["x_test"])
                    clf_metrics = compute_clf_metrics(y_true=data_dict["y_test"], y_clf=test_pred)

                    # Loop over cost matrixes
                    cost_matrix_list = get_cost_matrixes()
                    for cost_matrix in cost_matrix_list:
                        cost_mat_name = ";".join([str(x) for x in cost_matrix])

                        clf_value = compute_value(test_pred, data_dict["y_test"],
                                                  cost_matrix, REJECT_COST, None, NORMAL_TAG)
                        print("'%s - %s': accuracy %.4f, value %.3f" % (clf_name, cs_name, clf_metrics['acc'], clf_value))

                        # Loop over rejection strategies
                        rej_strategy_list = get_rejection_strategies(cost_matrix)
                        for rej_strategy in rej_strategy_list:
                            reject_name = rej_strategy.get_name()
                            reject_desc = rej_strategy.describe()

                            # Check if experiment was already executed
                            if existing_exps is not None and (((existing_exps['dataset_tag'] == dataset_name) &
                                                               (existing_exps['clf_name'] == clf_name) &
                                                               (existing_exps['calibration'] == cs_name) &
                                                               (existing_exps['cost_matrix'] == cost_mat_name) &
                                                               (existing_exps[
                                                                    'reject_strategy'] == reject_name)).any()):
                                print('%d/%d Skipping classifier %s, already in the results' % (
                                    exp_i, len(learners) * len(cost_matrix_list) * len(rej_strategy_list) * len(
                                        calibration_list), clf_name))

                            else:
                                # Otherwise we can move ahead
                                start_ms = current_ms()
                                rej_strategy.fit(proba=val_proba, y_pred=val_pred, y_true=data_dict["y_val"],
                                                 verbose=False)
                                rej_pred_y = rej_strategy.apply(test_proba=test_proba,
                                                                test_label=test_pred)
                                rej_time = current_ms() - start_time
                                value = compute_value(rej_pred_y, data_dict["y_test"],
                                                      cost_matrix, REJECT_COST, None, NORMAL_TAG)
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

                                # Updates CSV file with metrics of experiment
                                if not os.path.exists(SCORES_FILE):
                                    # Prints header
                                    with open(SCORES_FILE, 'w') as myfile:
                                        myfile.write("dataset_tag,clf_name,calibration,cost_matrix,reject_strategy,rej_desc,clf_value,")
                                        for met in clf_metrics:
                                            myfile.write(str(met) + ",")
                                        # Prints rej_clf stats
                                        myfile.write("rej_clf_value,")
                                        for met in rej_clf_metrics:
                                            myfile.write(str(met) + ",")
                                        # Prints rej_clf stats
                                        for met in rej_metrics:
                                            myfile.write(str(met) + ",")
                                        myfile.write("\n")
                                with open(SCORES_FILE, "a") as myfile:
                                    # Prints result of experiment in CSV file
                                    myfile.write("%s,%s,%s,%s,%s,%s," % (dataset_name, clf_name, cs_name, cost_mat_name, reject_name, reject_desc))
                                    # Prints clf stats
                                    myfile.write(str(clf_value) + ",")
                                    for met in clf_metrics:
                                        myfile.write(str(clf_metrics[met]) + ",")
                                    # Prints rej_clf stats
                                    myfile.write(str(value) + ",")
                                    for met in rej_clf_metrics:
                                        myfile.write(str(rej_clf_metrics[met]) + ",")
                                    # Prints rej_clf stats
                                    for met in rej_metrics:
                                        myfile.write(str(rej_metrics[met]) + ",")
                                    myfile.write("\n")

                            exp_i += 1

                classifier = None
