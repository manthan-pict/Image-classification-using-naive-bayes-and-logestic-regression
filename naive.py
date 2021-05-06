import numpy as np
from data import Data

mean_of_mean_trx0 = np.mean(Data.mean_trx0)
var_of_mean_trx0 = np.var(Data.mean_trx0)

mean_of_sd_trx0 = np.mean(Data.sd_trx0)
var_of_sd_Trx0 = np.var(Data.sd_trx0)

mean_of_mean_trx1 = np.mean(Data.mean_trx1)
var_of_mean_trx1 = np.var(Data.mean_trx1)

mean_of_sd_trx1 = np.mean(Data.sd_trx1)
var_sd_trx1 = np.var(Data.sd_trx1)


# Probability Density Function for 1 dimensional data
def pdf(mean, var, feature_value):
    numerator = np.exp(- (feature_value - mean) ** 2 / (2 * var))
    denominator = np.sqrt(2 * np.pi * var)
    return numerator / denominator


# Predicts class label for the given sample depending on the log likelyhood of the given sample.
def predict(x):
    log_of_prior = np.log(0.5)
    probability_sample_belongs_to_class0 = np.log(pdf(mean_of_sd_trx0, var_of_sd_Trx0, np.std(x))) + np.log(
        pdf(mean_of_mean_trx0, var_of_mean_trx0, np.mean(x))) + log_of_prior

    probability_sample_belongs_to_class1 = np.log(pdf(mean_of_sd_trx1, var_sd_trx1, np.std(x))) + np.log(
        pdf(mean_of_mean_trx1, var_of_mean_trx1, np.mean(x))) + log_of_prior

    if probability_sample_belongs_to_class0 < probability_sample_belongs_to_class1:
        return 1
    else:
        return 0


# Calculate the accuracy of the test dataset.
def calculate_accuracy(dataset, correct_label):
    correctly_classified_images = 0
    count = 0
    class_0_samples = 0
    class_1_samples = 0
    number_of_samples_in_class_0 = 0
    number_of_samples_in_class_1 = 0
    for x in dataset:
        class_label = predict(x)
        if correct_label[0, count] == 0:
            number_of_samples_in_class_0 += 1
        else:
            number_of_samples_in_class_1 += 1

        if class_label == correct_label[0, count]:
            correctly_classified_images += 1
            if class_label == 1:
                class_1_samples += 1
            else:
                class_0_samples += 1
        count += 1
    dataset_size = correct_label.size
    print("dataset_size: ", dataset_size)
    print("Combined Accuracy: ", (correctly_classified_images / dataset_size) * 100)
    print("Class 0 Accuracy: ", (class_0_samples / number_of_samples_in_class_0) * 100)
    print("Class 1 Accuracy: ", (class_1_samples / number_of_samples_in_class_1) * 100)


print("Training Data Details : ")
calculate_accuracy(Data.trX, Data.trY)

print("Testing Data Details : ")
calculate_accuracy(Data.tsX, Data.tsY)
