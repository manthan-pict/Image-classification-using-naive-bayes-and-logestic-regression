import numpy as np
import scipy.io


# trX - training set, each row represents a sample
# trY - training labels, 0 and 1 represent class 'tshirt' and class 'trouser' respectively
# tsX - testing set, each row represents a sample
# tsY - testing labels, 0 and 1 represent class 'tshirt' and class 'trouser' respectively

class Data:
    trX = scipy.io.loadmat('C:/Users/magraw12/PycharmProjects/cse575_project1/fashion_mnist.mat',
                           variable_names='trX').get('trX')
    trY = scipy.io.loadmat('C:/Users/magraw12/PycharmProjects/cse575_project1/fashion_mnist.mat',
                           variable_names='trY').get('trY')
    tsX = scipy.io.loadmat('C:/Users/magraw12/PycharmProjects/cse575_project1/fashion_mnist.mat',
                           variable_names='tsX').get('tsX')
    tsY = scipy.io.loadmat('C:/Users/magraw12/PycharmProjects/cse575_project1/fashion_mnist.mat',
                           variable_names='tsY').get('tsY')

    # Feature Extraction of samples in class0 - mean and standard deviation is computed on each image.
    trx0 = trX[0:6000]
    mean_trx0 = np.mean(trx0, axis=1)
    sd_trx0 = np.std(trx0, axis=1)

    # Feature Extraction of samples in class1 - mean and standard deviation is computed on each image.
    trx1 = trX[6000:12000]
    mean_trx1 = np.mean(trx1, axis=1)
    sd_trx1 = np.std(trx1, axis=1)