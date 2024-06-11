import configparser
import os
import shutil
import time

import numpy
import sklearn


def current_ms():
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)


def clean_name(file, prequel):
    """
    Method to get clean name of a file
    :param file: the original file path
    :return: the filename with no path and extension
    """
    if prequel in file:
        file = file.replace(prequel, "")
    if '.' in file:
        file = file.split('.')[0]
    if file.startswith("/"):
        file = file[1:]
    return file


def get_full_class_name(class_obj):
    return class_obj.__module__ + "." + class_obj.__qualname__


def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_classifier_name(classifier) -> str:
    """
    Gets a string with classifier name
    :return: a string
    """
    clf_name = classifier.classifier_name() if hasattr(classifier,
                                                       'classifier_name') else classifier.__class__.__name__
    if clf_name == 'Pipeline':
        keys = list(classifier.named_steps.keys())
        clf_name = str(keys) if len(keys) != 2 else str(keys[1]).upper()
    return clf_name


def compute_clf_metrics(y_true, y_clf, avoid_tag=None):
    """
    Computes metrics for a normal classifier
    :param avoid_tag: prediction tag to avoid when computing metrics
    :param y_true: the ground truth labels
    :param y_clf: the prediction of the regular classifier
    :return: a dictionary of metrics
    """
    to_avoid = []
    if avoid_tag in y_clf:
        to_avoid = 1*(y_clf == avoid_tag)
        pred_label = [i for i in y_clf if i != avoid_tag]
        true_label = numpy.asarray(y_true)
        true_label = true_label[to_avoid == 0]
    else:
        true_label = y_true
        pred_label = y_clf
    if len(true_label) > 0:
        met_dict = {'avoid': sum(to_avoid),
                    'acc': sklearn.metrics.accuracy_score(true_label, pred_label),
                    'mcc': sklearn.metrics.matthews_corrcoef(true_label, pred_label),
                    'b_acc': sklearn.metrics.balanced_accuracy_score(true_label, pred_label)}
        if len(numpy.unique(true_label)) == 1:
            # When the classifier predicts always the same class (shouldnt happen)
            met_dict['tn'] = sum(pred_label == true_label)
            met_dict['tp'] = 0
            met_dict['fn'] = sum(pred_label != true_label)
            met_dict['fp'] = 0
            met_dict['rec'] = 1.0
            met_dict['prec'] = 0.0
        elif len(numpy.unique(true_label)) == 2:
            # Metrics for binary classification
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(true_label, pred_label).ravel()
            met_dict['tn'] = tn
            met_dict['tp'] = tp
            met_dict['fn'] = fn
            met_dict['fp'] = fp
            met_dict['rec'] = sklearn.metrics.recall_score(true_label, pred_label)
            met_dict['prec'] = sklearn.metrics.precision_score(true_label, pred_label)
    else:
        met_dict = {'avoid': sum(to_avoid),
                    'acc': 0, 'mcc': 0, 'b_acc': 0,
                    'tn': 0, 'tp': 0, 'fn': 0, 'fp': 0, 'rec': 0, 'prec': 0}
    return met_dict


def compute_rejection_metrics(y_true, y_wrapper, y_clf, reject_tag=None):
    """
    Assumes that y_clf may have omissions, labeled as 'reject_tag'
    :param y_true: the ground truth labels
    :param y_wrapper: the prediction of the SPROUT (wrapper) classifier
    :param y_clf: the prediction of the regular classifier
    :param reject_tag: the tag used to label rejections, default is None
    :return: a dictionary of metrics
    """
    met_dict = {}
    met_dict['alpha'] = sklearn.metrics.accuracy_score(y_true, y_clf)
    met_dict['eps'] = 1 - met_dict['alpha']
    met_dict['phi'] = numpy.count_nonzero(y_wrapper == reject_tag) / len(y_true)
    met_dict['alpha_w'] = sum(y_true == y_wrapper) / len(y_true)
    met_dict['eps_w'] = 1 - met_dict['alpha_w'] - met_dict['phi']
    met_dict['phi_c'] = sum(numpy.where((y_wrapper == reject_tag) & (y_clf == y_true), 1, 0))/len(y_true)
    met_dict['phi_m'] = sum(numpy.where((y_wrapper == reject_tag) & (y_clf != y_true), 1, 0)) / len(y_true)
    met_dict['eps_gain'] = 0 if met_dict['eps'] == 0 else (met_dict['eps'] - met_dict['eps_w']) / met_dict['eps']
    met_dict['phi_m_ratio'] = 0 if met_dict['phi'] == 0 else met_dict['phi_m'] / met_dict['phi']
    return met_dict
