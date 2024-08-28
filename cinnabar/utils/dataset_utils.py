import copy
import os
import shutil
import urllib

import numpy
import pandas
import sklearn
from sklearn import datasets


def load_DIGITS(row_limit: int = numpy.nan, as_pandas: bool = False):
    """
    Loads DIGITS dataset from SKLearn
    :param row_limit: int (number of data points) if you want to use a portion of the dataset
    :param as_pandas: True if output has to be a Pandas Dataframe
    :return: features and labels with train/test split, label names and feature names
    """
    return process_image_dataset("DIGITS", limit=row_limit, as_pandas=as_pandas)


def load_MNIST(row_limit: int = numpy.nan, as_pandas: bool = False, flatten: bool = False):
    """
    Loads MNIST dataset
    :param flatten: True if dataset should be linearized
    :param row_limit: int (number of data points) if you want to use a portion of the dataset
    :param as_pandas: True if output has to be a Pandas Dataframe
    :return: features and labels with train/test split, label names and feature names
    """
    return process_image_dataset("MNIST", limit=row_limit, as_pandas=as_pandas, flatten=flatten)


def load_FASHIONMNIST(row_limit: int = numpy.nan, as_pandas: bool = False, flatten: bool = True):
    """
    Loads FASHION-MNIST dataset
    :param flatten: True if dataset should be linearized
    :param row_limit: int (number of data points) if you want to use a portion of the dataset
    :param as_pandas: True if output has to be a Pandas Dataframe
    :return: features and labels with train/test split, label names and feature names
    """
    return process_image_dataset("FASHION-MNIST", limit=row_limit, as_pandas=as_pandas, flatten=flatten)


def process_image_dataset(dataset_name: str, limit: int = numpy.nan, as_pandas: bool = False, flatten: bool = True):
    """
    Gets data for analysis, provided that the dataset is an image dataset
    :param flatten: True if dataset should be linearized
    :param as_pandas: True if output has to be a Pandas Dataframe
    :param dataset_name: name of the image dataset
    :param limit: specifies if the number of data points has to be cropped somehow (testing purposes)
    :return: many values for analysis
    """
    if dataset_name == "DIGITS":
        mn = datasets.load_digits(as_frame=True)
        feature_list = mn.feature_names
        labels = mn.target_names
        x_digits = mn.data
        y_digits = mn.target
        if (numpy.isfinite(limit)) & (limit < len(y_digits)):
            x_digits = x_digits[0:limit]
            y_digits = y_digits[0:limit]
        x_tr, x_te, y_tr, y_te = sklearn.model_selection.train_test_split(x_digits, y_digits, test_size=0.2,
                                                                          shuffle=True)
        return x_tr, x_te, y_tr, y_te, labels, feature_list

    elif dataset_name == "MNIST":
        mnist_folder = "input_folder/mnist"
        if not os.path.isdir(mnist_folder):
            print("Downloading MNIST ...")
            os.makedirs(mnist_folder)
            download_file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                          "train-images-idx3-ubyte.gz", mnist_folder)
            download_file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                          "train-labels-idx1-ubyte.gz", mnist_folder)
            download_file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                          "t10k-images-idx3-ubyte.gz", mnist_folder)
            download_file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                          "t10k-labels-idx1-ubyte.gz", mnist_folder)
        return format_mnist(mnist_folder, limit, as_pandas, flatten)

    elif dataset_name == "FASHION-MNIST":
        f_mnist_folder = "input_folder/fashion"
        if not os.path.isdir(f_mnist_folder):
            print("Downloading FASHION-MNIST ...")
            os.makedirs(f_mnist_folder)
            download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
                          "train-images-idx3-ubyte.gz", f_mnist_folder)
            download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
                          "train-labels-idx1-ubyte.gz", f_mnist_folder)
            download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
                          "t10k-images-idx3-ubyte.gz", f_mnist_folder)
            download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
                          "t10k-labels-idx1-ubyte.gz", f_mnist_folder)
        return format_mnist(f_mnist_folder, limit, as_pandas, flatten)


def download_file(file_url, file_name, folder_name):
    with urllib.request.urlopen(file_url) as response, open(folder_name + "/" + file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)


def read_tabular_dataset(dataset_name: str, label_name: str, limit: int = numpy.NaN, train_size: float = 0.5,
                         val_size: float = 0.2, shuffle: bool = True, l_encoding: bool = True) -> dict:
    """
    Method to process an input dataset as CSV
    :param l_encoding: if True, encodes labels as integers (useful for compatibility with some classifiers)
    :param shuffle: true if data has to be shuffled before splitting
    :param val_size: percentage of dataset to be used for validation
    :param train_size: percentage of dataset to be used for training
    :param limit: integer to cut dataset if needed.
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return: many values for analysis
    """
    # Loading Dataset
    df = pandas.read_csv(dataset_name, sep=",")

    # Shuffle
    if shuffle:
        df = df.sample(frac=1.0)
    df = df.fillna(0)
    df = df.replace('null', 0)

    # Testing Purposes
    if (numpy.isfinite(limit)) & (limit < len(df.index)):
        df = df[0:limit]

    if l_encoding:
        encoding = pandas.factorize(df[label_name])
        y_enc = encoding[0]
        labels = encoding[1]
    else:
        y_enc = df[label_name]
        labels = numpy.unique(y_enc)

    # Basic Pre-Processing
    print("\nDataset %s loaded: %d items" % (dataset_name, len(df.index)))

    # Train/Test Split of Classifiers
    x = df.drop(columns=[label_name])
    x_no_cat = x.select_dtypes(exclude=['object'])
    feature_list = x_no_cat.columns
    x_no_cat = x_no_cat.to_numpy()
    x_tr, x_te, y_tr, y_te = sklearn.model_selection.train_test_split(x_no_cat, y_enc,
                                                                      test_size=1 - train_size,
                                                                      shuffle=shuffle)
    if val_size > 0:
        x_val, x_te, y_val, y_te = sklearn.model_selection.train_test_split(x_te, y_te,
                                                                            test_size=1 - (val_size / (1 - train_size)),
                                                                            shuffle=shuffle)
    else:
        x_val = None
        y_val = None

    return {"x_train": x_tr, "x_test": x_te, "x_val": x_val, "y_train": y_tr, "y_test": y_te, "y_val": y_val,
            "label_names": labels, "feature_names": feature_list}


def read_binary_tabular_dataset(dataset_name: str, label_name: str, limit: int = numpy.NaN, train_size: float = 0.5,
                                val_size: float = 0.2, shuffle: bool = True, l_encoding: bool = False, normal_tag: str = 'normal') -> dict:
    """
    Method to process an input dataset as CSV
    :param normal_tag: string that identifies the class that has to be treated as normal. All other classes will become "anomaly"
    :param l_encoding: if True, encodes labels as integers (useful for compatibility with some classifiers)
    :param shuffle: true if data has to be shuffled before splitting
    :param val_size: percentage of dataset to be used for validation
    :param train_size: percentage of dataset to be used for training
    :param limit: integer to cut dataset if needed.
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return: many values for analysis
    """
    # Loading Dataset
    tab_dict = read_tabular_dataset(dataset_name, label_name, limit, train_size, val_size, shuffle, False)
    tab_dict["normal_perc"] = numpy.average(numpy.where(tab_dict["y_train"] == normal_tag, 1, 0))
    if l_encoding:
        tab_dict["y_train"] = numpy.where(tab_dict["y_train"] == normal_tag, 0, 1)
        tab_dict["y_test"] = numpy.where(tab_dict["y_test"] == normal_tag, 0, 1)
        if tab_dict["y_test"] is not None:
            tab_dict["y_val"] = numpy.where(tab_dict["y_val"] == normal_tag, 0, 1)
        tab_dict["label_names"] = [0, 1]
    else:
        tab_dict["y_train"] = numpy.where(tab_dict["y_train"] == normal_tag, normal_tag, "anomaly")
        tab_dict["y_test"] = numpy.where(tab_dict["y_test"] == normal_tag, normal_tag, "anomaly")
        if tab_dict["y_test"] is not None:
            tab_dict["y_val"] = numpy.where(tab_dict["y_val"] == normal_tag, normal_tag, "anomaly")
        tab_dict["label_names"] = [normal_tag, "anomaly"]
    return tab_dict


def read_unknown_tabular_dataset(dataset_name: str, label_name: str, limit: int = numpy.NaN, train_size: float = 0.5,
                                val_size: float = 0.2, shuffle: bool = True, l_encoding: bool = False, normal_tag: str = 'normal') -> dict:
    """
    Method to process an input dataset as CSV
    :param normal_tag: string that identifies the class that has to be treated as normal. All other classes will become "anomaly"
    :param l_encoding: if True, encodes labels as integers (useful for compatibility with some classifiers)
    :param shuffle: true if data has to be shuffled before splitting
    :param val_size: percentage of dataset to be used for validation
    :param train_size: percentage of dataset to be used for training
    :param limit: integer to cut dataset if needed.
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return: many values for analysis
    """
    # Loading Dataset
    tab_dict = read_tabular_dataset(dataset_name, label_name, limit, train_size, val_size, shuffle, False)
    class_names = tab_dict["label_names"]
    if len(class_names) > 2:
        arr_dict = {}
        for c_name in class_names:
            if c_name != normal_tag:
                tag = dataset_name + "@" + c_name
                train_indexes = tab_dict["y_train"] != c_name
                val_indexes = tab_dict["y_val"] != c_name
                test_indexes = tab_dict["y_test"] == c_name
                arr_dict[tag] = {"x_train": tab_dict["x_train"][train_indexes, :],
                                 "x_test_unk": tab_dict["x_test"][test_indexes, :],
                                 "x_test": tab_dict["x_test"],
                                 "x_val": tab_dict["x_val"][val_indexes, :],
                                 "y_train": numpy.where(tab_dict["y_train"][train_indexes] == normal_tag, 0, 1),
                                 "y_test_unk": numpy.where(tab_dict["y_test"][test_indexes] == normal_tag, 0, 1),
                                 "y_test": numpy.where(tab_dict["y_test"] == normal_tag, 0, 1),
                                 "y_val": numpy.where(tab_dict["y_val"][val_indexes] == normal_tag, 0, 1),
                                 "label_names": [0, 1],
                                 "feature_names": tab_dict["feature_names"]}

        return arr_dict
    else:
        return {}


def is_image_dataset(dataset_name) -> bool:
    """
    Checks if a dataset is an image dataset.
    :param dataset_name: name/path of the dataset
    :return: True if the dataset is not a tabular (CSV) dataset
    """
    return (dataset_name == "DIGITS") or (dataset_name != "MNIST") or (dataset_name != "FASHION-MNIST")


def load_mnist(path, kind='train'):
    """
    Taken from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    :param path: path where the mnist-like dataset is stored
    :param kind: to navigate between mnist-like archives
    :return: train/test set of the mnist-like dataset
    """
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
        images = images.reshape(len(labels), 28, 28)

    return images, labels


def format_mnist(mnist_folder, limit, as_pandas, flatten=True):
    """
    Loads an mnist-like dataset and provides as output the train/test split plus features
    :param as_pandas: True if output has to be a Pandas Dataframe
    :param mnist_folder: folder to load the mnist-like dataset
    :param limit: specifies if the number of data points has to be cropped somehow (testing purposes)
    :return: many values for analysis
    """
    x_tr, y_tr = load_mnist(mnist_folder, kind='train')
    x_te, y_te = load_mnist(mnist_folder, kind='t10k')

    # Linearizes features in the 28x28 image
    if flatten:
        x_tr = numpy.stack([x.flatten() for x in x_tr])
        x_te = numpy.stack([x.flatten() for x in x_te])
    x_fmnist = numpy.concatenate([x_tr, x_te], axis=0)
    y_fmnist = numpy.concatenate([y_tr, y_te], axis=0)

    # Crops if needed
    if (numpy.isfinite(limit)) & (limit < len(x_fmnist)):
        x_tr = x_tr[0:int(limit / 2)]
        y_tr = y_tr[0:int(limit / 2)]
        x_te = x_te[0:int(limit / 2)]
        y_te = y_te[0:int(limit / 2)]

    # Lists feature names and labels
    feature_list = ["pixel_" + str(i) for i in numpy.arange(0, len(x_fmnist[0]), 1)]
    labels = pandas.Index(numpy.unique(y_fmnist), dtype=object)

    if as_pandas:
        return pandas.DataFrame(data=x_tr, columns=feature_list), \
               pandas.DataFrame(data=x_te, columns=feature_list), \
               pandas.DataFrame(data=y_tr, columns=['label']), \
               pandas.DataFrame(data=y_te, columns=['label']), labels, feature_list
    else:
        return x_tr, x_te, y_tr, y_te, labels, feature_list
