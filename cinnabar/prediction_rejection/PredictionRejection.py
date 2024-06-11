import numpy
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


def compute_binary_value(y_pred, y_true, cost_matrix, reject_value: int = 0, reject_tag=None, normal_tag=0):
    # now lets compute the actual value of each prediction, initializing with value of TP, then updating FP, FN, TN
    value_vector = numpy.full(y_pred.shape[0], cost_matrix[0])
    value_vector[(y_pred != normal_tag) & (y_true != y_pred)] = cost_matrix[1]
    value_vector[(y_pred == normal_tag) & (y_true != y_pred)] = cost_matrix[2]
    value_vector[(y_pred == normal_tag) & (y_true == y_pred)] = cost_matrix[3]
    # loss due to rejects
    value_vector[y_pred == reject_tag] = reject_value
    # Final value
    value = numpy.sum(value_vector) / len(y_true)
    return value


def compute_multi_value(y_pred, y_true, cost_matrix, reject_value: int = 0, reject_tag=None):
    # now lets compute the actual value of each prediction, initializing with value of correct, then updating misc
    value_vector = numpy.full(y_pred.shape[0], cost_matrix[0])
    value_vector[(y_pred != y_true)] = cost_matrix[1]
    # loss due to rejects
    value_vector[y_pred == reject_tag] = reject_value
    # Final value
    value = numpy.sum(value_vector) / len(y_true)
    return value


def compute_value(y_pred, y_true, cost_matrix, reject_value: int = 0, reject_tag=None, normal_tag=0):
    """
    Function to compute the value of a prediction (with or without rejections)
    :param normal_tag: the tag to understand what the normal class is (used only for binary classification)
    :param y_pred: the prediction of the classifier
    :param y_true: the ground truth
    :param cost_matrix: the cost matrix
    :param reject_value: the value (cost) assigned to rejects, default is 0
    :param reject_tag: the tag to understand where a reject is
    :return:
    """
    if cost_matrix is None:
        return numpy.NaN
    if len(cost_matrix) == 4:
        return compute_binary_value(y_pred, y_true, cost_matrix, reject_value, reject_tag, normal_tag)
    else:
        return compute_binary_value(y_pred, y_true, cost_matrix, reject_value, reject_tag)


class PredictionRejection:
    """
    Class for building objects able to reject predictions according to specific rules
    """

    def __init__(self, cost_matrix, reject_cost: int = 0):
        """
        Constructor
        :param cost_matrix: the cost matrix (may not be used)
        :param reject_cost: the cost of rejection
        """
        self.cost_matrix = cost_matrix
        self.reject_cost = reject_cost

    def fit(self, proba: numpy.ndarray, y_pred: numpy.ndarray, y_true: numpy.ndarray, verbose=True):
        """
        Makes the prediction rejection strategy ready to be applied.
        In this case, it identifies ranges in which predictions should be excluded
        :return:
        """
        pass

    def is_binary(self) -> bool:
        """
        True if binary classification
        :return: boolean
        """
        return True if self.cost_matrix is not None and len(self.cost_matrix) == 4 else False

    def find_rejects(self, test_proba: numpy.ndarray, reject_ranges: list = None):
        """
        Findes rejections in a specific test set
        :param reject_ranges: ranges to be used for rejecting: if a proba is in range, is rejected
        :param test_proba: the data to apply the strategy to
        :return:
        """
        pass

    def is_fit(self) -> bool:
        """
        True if already fit
        :return: boolean
        """
        pass

    def apply(self, test_proba: numpy.ndarray, test_label: numpy.ndarray = None, reject_tag=None,
              reject_ranges: list = None):
        """
        Applies the prediction rejection strategy to a specific test set
        :param reject_ranges: ranges to be used for rejecting: if a proba is in range, is rejected
        :param test_label: the predicted labels
        :param reject_tag: the item that corresponds to a prediction rejection
        :param test_proba: the data to apply the strategy to
        :return:
        """
        if test_label is None:
            test_label = numpy.argmax(test_proba, axis=1)
        rejects = self.find_rejects(test_proba, reject_ranges)
        return numpy.asarray([reject_tag if rejects[i] > 0 else test_label[i] for i in range(len(rejects))])

    def get_name(self) -> str:
        """
        Returns the name of the strategy
        :return:
        """
        return self.__class__.__name__

    def describe(self) -> str:
        """
        Returns a textual description of the rejection strategy
        :return: a string
        """
        if hasattr(self, 'reject_ranges') and len(self.reject_ranges) > 0:
            if isinstance(self.reject_ranges[0], list):
                return "Reject thrs: " + ";".join([str(x[0]) + "-" + str(x[1]) for x in self.reject_ranges])
            else:
                return "Reject thrs: " + str(self.reject_ranges[0]) + "-" + str(self.reject_ranges[1])
        elif hasattr(self, 'strategy'):
            return self.strategy
        else:
            return ""


class ValueAware(PredictionRejection):
    """
    Binary value aware strategy to reject predictions
    From Sayin, B., Casati, F., Passerini, A., Yang, J., & Chen, X. (2022). Rethinking and recomputing the value of ML models. arXiv preprint arXiv:2209.15157
    """

    def __init__(self, cost_matrix, reject_cost: int = 0, candidate_thresholds: list = [0.5], normal_tag=0):
        """
        Constructor
        :param cost_matrix: the cost matrix (may not be used)
        :param val_proba: the probabilities assigned by a classifier to validation set
        :param data_dict: full dataset
        :param classifier: the classifier under investigation
        """
        PredictionRejection.__init__(self, cost_matrix=cost_matrix, reject_cost=reject_cost)
        self.reject_ranges = []
        self.candidate_thresholds = candidate_thresholds
        self.normal_tag = normal_tag

    def fit(self, proba: numpy.ndarray, y_pred: numpy.ndarray, y_true: numpy.ndarray, verbose=True):
        """
        Makes the prediction rejection strategy ready to be applied.
        In this case, it identifies ranges in which predictions should be excluded
        :return:
        """
        cost_list = {}
        if self.is_binary():
            # finding the FP thr
            best_fp = 0
            max_value = -numpy.inf
            for t_fp in self.candidate_thresholds:
                # here we define K = fn_c_norm, change it based on task.
                y_rej = self.apply(proba, None, None, [t_fp, 0.5])
                value = compute_binary_value(y_rej, y_true, self.cost_matrix, self.reject_cost, None, self.normal_tag)
                if value > max_value:
                    best_fp = t_fp
                    max_value = value
            # finding the FN thr
            best_fn = 0
            max_value = -numpy.inf
            for t_fn in self.candidate_thresholds:
                # here we define K = fn_c_norm, change it based on task.
                y_rej = self.apply(proba, None, None, [0.5, t_fn])
                value = compute_binary_value(y_rej, y_true, self.cost_matrix, self.reject_cost, None,
                                             self.normal_tag)
                if value > max_value:
                    best_fn = t_fn
                    max_value = value
            self.reject_ranges = [best_fp, best_fn]
        else:
            print("Multiclass\n")

    def is_fit(self) -> bool:
        """
        True if already fit
        :return: boolean
        """
        return self.reject_ranges is not None and len(self.reject_ranges) > 0

    def find_rejects(self, test_proba: numpy.ndarray, reject_ranges: list = None):
        """
        Findes rejections in a specific test set
        :param reject_ranges: ranges to be used for rejecting: if a proba is in range, is rejected
        :param test_proba: the data to apply the strategy to
        :return:
        """
        if reject_ranges is None:
            reject_ranges = self.reject_ranges
        rejects = numpy.full(test_proba.shape[0], 0)
        if self.is_binary():
            # the binary case has two thresholds, one for FP and one for FN
            rejects[test_proba[:, 1] < reject_ranges[0]] = 1
            rejects[test_proba[:, 0] < reject_ranges[1]] = 1
        else:
            # The multi-class case has a single threshold on argmax
            max_proba = numpy.argmax(test_proba, axis=0)
            rejects[max_proba < reject_ranges[0]] = 1
        return rejects


class SufficientlySafe(PredictionRejection):
    """
    Binary value aware strategy to reject unsafe predictions
    From Gharib, M., Zoppi, T., & Bondavalli, A. (2022). On the properness of incorporating binary classification machine learning algorithms into safety-critical systems. IEEE Transactions on Emerging Topics in Computing, 10(4), 1671-1686.
    """

    def __init__(self, cost_matrix, reject_cost: int = 0, max_iterations: int = 5):
        """
        Constructor
        :param cost_matrix: the cost matrix (may not be used)
        :param val_proba: the probabilities assigned by a classifier to validation set
        :param data_dict: full dataset
        :param classifier: the classifier under investigation
        """
        PredictionRejection.__init__(self, cost_matrix=cost_matrix, reject_cost=reject_cost)
        self.reject_ranges = []
        self.max_iterations = max_iterations
        # The ALR is the acceptable level or risk (acceptable amount of FNs)
        # It depends on the cost associated to the FNs, here is ALR = abs(1 / cost(FN))
        # The bigger the cost, the lower the ALR
        self.alr = abs(1 / cost_matrix[2])

    def fit(self, proba: numpy.ndarray, y_pred: numpy.ndarray, y_true: numpy.ndarray, verbose=True):
        """
        Makes the prediction rejection strategy ready to be applied.
        In this case, it identifies ranges in which predictions should be excluded
        :return:
        """
        fns = (y_true != y_pred) * (proba[:, 0] > 0.5)
        fn_probas = sorted(proba[fns, 0])
        i = 0
        while len(fn_probas) > 0:
            residual_fns = self.residual_fns(proba, fns, self.reject_ranges)
            if residual_fns < self.alr:
                # Means that we are able to avoid enough FNs to comply with the ALR
                break
            # Otherwise, we have to modify the reject range (reject more)
            self.reject_ranges = [[0.5, fn_probas.pop(0)]]
            i += 1
        if verbose:
            print("Ended with reject range %s and SSPr=%.3f" % (";".join([str(x) for x in self.reject_ranges]),
                                                                self.sufficiently_safe_value(proba)))

    def is_fit(self) -> bool:
        """
        True if already fit
        :return: boolean
        """
        return self.reject_ranges is not None and len(self.reject_ranges) > 0

    def residual_fns(self, test_proba: numpy.ndarray, fn_list: numpy.ndarray, reject_ranges: list = None) -> float:
        """
        Returns the % of FNs that are not rejected
        :param test_proba: the probabilities on some test set
        :param fn_list: the list of false negatives
        :param reject_ranges: (optional) custom reject ranges, or self.reject_ranges are used instead
        :return: a percentage (to be compared with ALR)
        """
        fn_count = sum(1 * fn_list)
        rejects = self.find_rejects(test_proba, reject_ranges)
        rejected_fn = rejects * fn_list
        return (fn_count - sum(rejected_fn)) / len(fn_list)

    def sufficiently_safe_value(self, test_proba: numpy.ndarray, reject_ranges: list = None) -> float:
        """
        returns the SSPr value of the paper
        :param test_proba: probability of the test set
        :param reject_ranges: (optional) custom reject ranges, or self.reject_ranges are used instead
        :return: SSPr value
        """
        rejects = self.find_rejects(test_proba, reject_ranges)
        nssp = sum(rejects)
        ssp = len(rejects) - nssp
        return ssp / (nssp + ssp)

    def find_rejects(self, test_proba: numpy.ndarray, reject_ranges: list = None):
        """
        Findes rejections in a specific test set
        :param reject_ranges: ranges to be used for rejecting: if a proba is in range, is rejected
        :param test_proba: the data to apply the strategy to
        :return:
        """
        if reject_ranges is None:
            reject_ranges = self.reject_ranges
        if len(reject_ranges) == 0:
            reject_ranges = [[-1, -1], [2, 2]]
        into_intervals = [numpy.where((my_range[0] <= test_proba[:, 0]) & (test_proba[:, 0] <= my_range[1]), 1, 0) for
                          my_range in reject_ranges]
        into_i = numpy.sum(numpy.asarray(into_intervals), axis=0)
        rejects = 1 * (into_i > 0)
        return rejects

    def get_name(self) -> str:
        """
        Returns the name of the strategy
        :return:
        """
        return self.__class__.__name__ + "(" + str(self.alr) + ")"


class EntropyRejection(PredictionRejection):
    """
    Computes entropy in probabilities and use them for training a reject classifier
    """

    def __init__(self, cost_matrix, classifier=DecisionTreeClassifier(), reject_cost: int = 0):
        """
        Constructor
        :param cost_matrix: the cost matrix (may not be used)
        :param val_proba: the probabilities assigned by a classifier to validation set
        :param data_dict: full dataset
        :param classifier: the classifier under investigation
        """
        PredictionRejection.__init__(self, cost_matrix=cost_matrix, reject_cost=reject_cost)
        self.classifier = classifier

    def fit(self, proba: numpy.ndarray, y_pred: numpy.ndarray, y_true: numpy.ndarray, verbose=True):
        """
        Makes the prediction rejection strategy ready to be applied.
        In this case, it identifies ranges in which predictions should be excluded
        :return:
        """
        x_train = self.get_entropy(proba)
        y_train = 1*(y_pred != y_true)
        self.classifier.fit(x_train, y_train)

    def get_entropy(self, proba):
        """
        Constructor Method
        :param proba: probabilities to compute entropy and max of
        """
        norm_array = numpy.full(proba.shape[1], 1 / proba.shape[1])
        with numpy.errstate(divide='ignore', invalid='ignore'):
            normalization = (-norm_array * numpy.log2(norm_array)).sum()
            entropy = numpy.sum(-proba * numpy.log2(proba), axis=1)
        ents = (normalization - entropy) / normalization
        max_p = numpy.max(proba, axis=1)
        return numpy.vstack([ents, max_p]).T

    def is_fit(self) -> bool:
        """
        True if already fit
        :return: boolean
        """
        try:
            if self.classifier is None:
                return False
            check_is_fitted(self.classifier)
            return True
        except NotFittedError:
            return False

    def find_rejects(self, test_proba: numpy.ndarray, reject_ranges: list = None):
        """
        Findes rejections in a specific test set
        :param reject_ranges: ranges to be used for rejecting: if a proba is in range, is rejected
        :param test_proba: the data to apply the strategy to
        :return:
        """
        x_test = self.get_entropy(test_proba)
        return self.classifier.predict(x_test)


class EnsembleRejection(PredictionRejection):
    """
    Computes many measures to understand when to reject
    """

    def __init__(self, cost_matrix, reject_cost: int = 0, rejectors: list = [], strategy: str = 'all'):
        """
        Constructor
        :param cost_matrix: the cost matrix (may not be used)
        """
        PredictionRejection.__init__(self, cost_matrix=cost_matrix, reject_cost=reject_cost)
        self.rejectors = rejectors
        self.strategy = strategy

    def fit(self, proba: numpy.ndarray, y_pred: numpy.ndarray, y_true: numpy.ndarray, verbose=True):
        """
        Makes the prediction rejection strategy ready to be applied.
        In this case, it identifies ranges in which predictions should be excluded
        :return:
        """
        for rejector in self.rejectors:
            if not rejector.is_fit():
                rejector.fit(proba, y_pred, y_true, verbose)

    def is_fit(self) -> bool:
        """
        :return:
        """
        for rejector in self.rejectors:
            if not rejector.is_fit():
                return False
        return True


    def find_rejects(self, test_proba: numpy.ndarray, reject_ranges: list = None):
        """
        Finds rejections in a specific test set
        :param reject_ranges: ranges to be used for rejecting: if a proba is in range, is rejected
        :param test_proba: the data to apply the strategy to
        :return:
        """
        rejs = []
        for rejector in self.rejectors:
            rejs.append(rejector.find_rejects(test_proba, None))
        rejs = numpy.sum(numpy.vstack(rejs), axis=0)
        if self.strategy == 'one':
            return numpy.where(rejs > 0, 1, 0)
        elif self.strategy == 'two':
            return numpy.where(rejs > 1, 1, 0)
        elif self.strategy == 'all':
            return numpy.where(rejs == len(self.rejectors), 1, 0)
        else:
            return numpy.zeros(test_proba.shape[0])

    def get_name(self) -> str:
        """
        Returns the name of the strategy
        :return:
        """
        return self.__class__.__name__ + "(" + str(self.strategy) + ")"
