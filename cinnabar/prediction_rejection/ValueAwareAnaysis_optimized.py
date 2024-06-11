import numpy as np


# ALL

def cost_based_threshold(k):
    t = (k) / (k + 1)
    return t


class CostMetric:

    def __init__(self, logits_val, y_val, logits_test, y_test):
        self.logits_val = logits_val
        self.y_val = y_val
        self.logits_test = logits_test
        self.y_test = y_test

    def cost_based_analysis(self, Vr, Vc, Vw, confT_list):
        pass


class BinaryCostMetric(CostMetric):

    def __init__(self, logits_val, y_val, logits_test, y_test):
        super().__init__(logits_val, y_val, logits_test, y_test)

    def calculate_value(self, y_hat_proba, y, t_fp, V_fp, t_fn, V_fn, Vc, Vr):
        """ calculate value of classificator

        Args:
            y_hat_proba (2D npy array of float): contains confidences score on the set
            y (1D npy array of 0 or 1): contains ground truth of the set
            t_fp (float): thresold for false positive
            V_fp (float): value of FP
            t_fn (float): thresold for false negative
            V_fn (float): value of FN
            Vc (float): value of correct classification
            Vr (float): value of reject classification

        Returns:
            float: value of classificator
            int: number of rejected samples
            int: number of wrong predictions
            int: number of correct predictions

        """

        values = [Vc, V_fp, V_fn]
        n_samples = len(y)
        value_vector = np.full(n_samples, Vr)

        # if any threshold is below 0.5 we need to make an extra check to assure that we are considering the most confident prediction
        if ((t_fp < 0.5) or (t_fn < 0.5)):
            # conditions to decide the value of the prediction
            cond1 = (((y == 1) & (y_hat_proba[:, 1] > t_fp) & (y_hat_proba[:, 1] > y_hat_proba[:, 0])) | (
                        (y == 0) & (y_hat_proba[:, 0] > t_fn)) & (y_hat_proba[:, 0] > y_hat_proba[:, 1]))
            cond2 = (y_hat_proba[:, 1] > y_hat_proba[:, 0]) & (y != 1) & (y_hat_proba[:, 1] > t_fp)
            cond3 = (y_hat_proba[:, 0] > y_hat_proba[:, 1]) & (y != 0) & (y_hat_proba[:, 0] > t_fn)

            # Assigns the correct value to each prediction
            value_vector[cond1] = values[0]
            value_vector[cond2] = values[1]
            value_vector[cond3] = values[2]

        else:
            # conditions to decide the value of the prediction
            cond1 = ((y == 1) & (y_hat_proba[:, 1] > t_fp)) | ((y == 0) & (y_hat_proba[:, 0] > t_fn))
            cond2 = (y != 1) & (y_hat_proba[:, 1] > t_fp)
            cond3 = (y != 0) & (y_hat_proba[:, 0] > t_fn)

            # Assigns the correct value to each prediction
            value_vector[cond1] = values[0]
            value_vector[cond2] = values[1]
            value_vector[cond3] = values[2]

        # Calculate the total value
        value = np.sum(value_vector) / n_samples

        # Calculate the number of rejected samples, wrong predictions, and correct predictions
        numOfWrongPredictions = len(value_vector[cond2]) + len(value_vector[cond3])
        numOfCorrectPredictions = len(value_vector[cond1])
        numOfRejectedSamples = n_samples - numOfCorrectPredictions - numOfWrongPredictions

        return value, numOfRejectedSamples, numOfWrongPredictions, numOfCorrectPredictions

    def calculate_value_without_rejection(self, y_hat_proba, y, V_fp, V_fn, Vc):
        """ calculate value of classificator assuming rejection is not allowed

        Args:
            y_hat_proba (2D npy array of float): contains confidences score on the set
            y (1D npy array of 0 or 1): contains ground truth of the set
            V_fp (float): value of FP
            V_fn (float): value of FN
            Vc (float): value of correct classification

        Returns:
            float: value of classificator
            int: number of wrong samples
            int: number of correct predictions

        """
        values = [V_fp, V_fn]
        n_samples = len(y)
        value_vector = np.full(n_samples, Vc)

        # conditions to decide the value of the prediction
        cond1 = (y != 1) & (y_hat_proba[:, 1] > y_hat_proba[:, 0])
        cond2 = (y != 0) & (y_hat_proba[:, 0] > y_hat_proba[:, 1])

        # Assigns the correct value to each prediction
        value_vector[cond1] = values[0]
        value_vector[cond2] = values[1]

        # Calculate the total value
        value = np.sum(value_vector) / n_samples

        # Calculate the number of wrong predictions and correct predictions
        numOfWrongPredictions = len(value_vector[cond1]) + len(value_vector[cond2])
        numOfCorrectPredictions = n_samples - numOfWrongPredictions

        return value, numOfWrongPredictions, numOfCorrectPredictions

    def find_optimum_confidence_threshold_fp(self, y_hat_proba, y, theoretical, Vw_fp, Vc, Vr, confidence, precision):
        """ calculates the best empirical t_fp given a precision and a confidence

        Args:
            y_hat_proba (2D npy array of float): contains confidences score on the set
            y (1D npy array of 0 or 1): contains ground truth of the set
            theoretical (float): contains the middle point of the search, good estimate would be the
                                 theoretical threshold
            Vw_fp (float): value of FP
            Vc (float): value of correct classification
            confidence (float): how far from the middle point to search, 1 is equal to searching the whole [0,1] range,
                                should be positive
            precision (int): number of decimal numbers of the step to iterate inside the [0,1] range,
                             1 means a step of 0.1, should be positive
        Returns:
            float: best empirical t_fp given a precision and a confidence

        """
        # initialize
        max_value = float('-inf')
        max_t_fp = 0
        step = 10 ** -precision
        # rounds to be consistent
        theoretical = round(theoretical, precision)
        # iterates over all values of t_fp within a confidence range of the theoretical threshold with a step
        for t_fp in np.arange(max(theoretical - confidence, 0), min(theoretical + confidence + step, 1 + step), step):
            value, _, _, _ = self.calculate_value(y_hat_proba, y, t_fp, Vw_fp, 0.6, 0, Vc, Vr)
            # if value is higher select new best threshold
            if (max_value < value):
                max_value = value
                max_t_fp = t_fp
        return max_t_fp

    def find_optimum_confidence_threshold_fn(self, y_hat_proba, y, theoretical, Vw_fn, Vc, Vr, confidence, precision):
        """ calculates the best empirical t_fn given a precision and a confidence

        Args:
            y_hat_proba (2D npy array of float): contains confidences score on the set
            y (1D npy array of 0 or 1): contains ground truth of the set
            theoretical (float): contains the middle point of the search, good estimate would be the
                                 theoretical threshold
            Vw_fn (float): value of FN
            Vc (float): value of correct classification
            confidence (float): how far from the middle point to search, 1 is equal to searching the whole [0,1] range,
                                should be positive
            precision (int): number of decimal numbers of the step to iterate inside the [0,1] range,
                             1 means a step of 0.1, should be positive
        Returns:
            float: best empirical t_fn given a precision and a confidence

        """
        # initialize
        max_value = float('-inf')
        max_t_fn = 0
        step = 10 ** -precision
        # rounds to be consistent
        theoretical = round(theoretical, precision)
        # iterates over all values of t_fn within a confidence range of the theoretical threshold with a step
        for t_fn in np.arange(max(theoretical - confidence, 0), min(theoretical + confidence + step, 1 + step), step):
            value, _, _, _ = self.calculate_value(y_hat_proba, y, 0.6, 0, t_fn, Vw_fn, Vc, Vr)
            # if value is higher select new best threshold
            if (max_value < value):
                max_value = value
                max_t_fn = t_fn
        return max_t_fn

    def cost_based_analysis(self,
            y_hat_proba_val,
            y_val,
            y_hat_proba_test,
            y_test,
            res_path,
            logfile_name,
            Vr,
            Vc,
            Vw_list_fp,
            Vw_list_fn,
            precision_fp,
            precision_fn,
            confidence_fp,
            confidence_fn,
    ):
        """ creates a file containing the value of a series of predictions using both a theoretical
        and an empirically found perfect threshold for all the different error costs combinations

        Args:
            y_hat_proba_val (2D npy array of float): contains confidences score on the validation set
            y_val (1D npy array of 0 or 1): contains ground truth of the validation set
            y_hat_proba_test (2D npy array of float): contains confidences score on the test set
            y_test (1D npy array of 0 or 1): contains ground truth of the test set
            res_path (str): directory in which to print the file
            logfile_name (str): name of result file
            Vc (float): value of correct classification
            Vr (float): value of reject classification
            Vw_list_fp (list of float): list of values of FP error
            Vw_list_fn (list of float): list of values of FN error
            precision_fp (int): precision to use when searching for empirical threshold fp
            precision_fn (int): precision to use when searching for empirical threshold fn
            confidence_fp (float): confidence to use when searching for empirical threshold fp
            confidence_fn (float): confidence to use when searching for empirical threshold fn
        """

        # pre-compute all theoretical thresholds for each FN value
        fn_list = []
        for Vw_fn in Vw_list_fn:
            # calculate theoretical threshold
            k_fn = (-1) * (Vw_fn / Vc)
            t_fn = cost_based_threshold(k_fn)
            # calculate empirically perfect threshold
            fn_list.append((self.find_optimum_confidence_threshold_fn(
                y_hat_proba_val, y_val, t_fn, Vw_fn, Vc, Vr, confidence_fn, precision_fn
            ), t_fn, Vw_fn))

        # iterates over all possible values for FP
        for Vw_fp in Vw_list_fp:
            data_log = []

            # calculate theoretical threshold
            k_fp = (-1) * (Vw_fp / Vc)
            t_fp = cost_based_threshold(k_fp)

            # calculate empirically perfect threshold
            e_fp = self.find_optimum_confidence_threshold_fp(
                y_hat_proba_val, y_val, t_fp, Vw_fp, Vc, Vr, confidence_fp, precision_fp
            )

            for e_fn, t_fn, Vw_fn in fn_list:
                # calculate value using theoretical best threshold
                value_test, rej_test, wrong_test, correct_test = self.calculate_value(
                    y_hat_proba_test, y_test, e_fp, Vw_fp, t_fn, Vw_fn, Vc, Vr
                )

                # calculate value using empirical perfect threshold
                value_test_opt, rej_test_opt, wrong_test_opt, correct_test_opt = self.calculate_value(
                    y_hat_proba_test,
                    y_test,
                    e_fp,
                    Vw_fp,
                    e_fn,
                    Vw_fn,
                    Vc,
                    Vr,
                )

                # calculate value assuming no rejection
                value_test_no_rej, wrong_test_no_rej, correct_test_no_rej = self.calculate_value_without_rejection(
                    y_hat_proba_test, y_test, Vw_fp, Vw_fn, Vc
                )

                # calculate theoretical threshold
                k_fn = (-1) * (Vw_fn / Vc)

                # handles output to file
                data_log.append(
                    f"{Vr},{Vc},{Vw_fp},{Vw_fn},{k_fp},{k_fn},{t_fp},{t_fn},{value_test},{rej_test},{wrong_test},{correct_test},{e_fp},{e_fn},{value_test_opt},{rej_test_opt},{wrong_test_opt},{correct_test_opt},{value_test_no_rej},{wrong_test_no_rej},{correct_test_no_rej}\n"
                )


# MULTI

class MultiClassCostMetric(CostMetric):

    def __init__(self, logits_val, y_val, logits_test, y_test):
        super().__init__(logits_val, y_val, logits_test, y_test)


