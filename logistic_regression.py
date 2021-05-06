import numpy as np
from data import Data

# Created feature vector of the training Sample data.
mean_of_trx = np.array(np.append(Data.mean_trx1, Data.mean_trx0))
sd_of_trx = np.array(np.append(Data.sd_trx1, Data.sd_trx0))


# Converts the given linear value into probability values between zero and one.
def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))


# Calculates the probability values for the given weights and feature vectors.
def probability_of_sample(weights, feature_mean, feature_standard_deviation):
    linear_value = np.array(
        weights[0] + weights[1] * feature_mean + weights[2] * feature_standard_deviation)
    return sigmoid_function(linear_value)


# Log likelihood function to calculate the cost.
def cost(weights, feature_mean, feature_standard_deviation, true_class_for_samples):
    y_pred = probability_of_sample(weights, feature_mean, feature_standard_deviation)
    return sum(true_class_for_samples * np.log(y_pred) + (1 - true_class_for_samples) * np.log(
        1 - y_pred + pow(1, np.exp(-8))))


# Calculates the gradient values using weights,feature vector and class values of samples.
def grad(w, feature_mean, feature_standard_deviation, label):
    y_pred = probability_of_sample(w, feature_mean, feature_standard_deviation)
    g = [0] * 3
    g[0] = sum(label - y_pred)
    g[1] = sum((label - y_pred) * feature_mean)
    g[2] = sum((label - y_pred) * feature_standard_deviation)
    return g


# Gradient assent function to update the weights using learning rate
def assent(w_new, w_prev, lr):
    # Commented out to reduce extra info printed on console.
    # print(w_prev)
    # print(cost(w_prev, feature_vector, Data.trY[0]))
    j = 0
    while True:
        w_prev = w_new
        gradient = grad(w_prev, mean_of_trx, sd_of_trx, Data.trY[0])
        w0 = w_prev[0] + lr * gradient[0]
        w1 = w_prev[1] + lr * gradient[1]
        w2 = w_prev[2] + lr * gradient[2]
        w_new = [w0, w1, w2]
        # Commented out to reduce extra info printed on console.
        # print(w_new)
        # print(cost(w_new, feature_vector, Data.trY[0]))

        if j > 1000:
            return w_new
        j += 1


# takes the input dataset and predict the class labels for the samples using given weights.
# Then labels will be compared to actual class labels and accuracy is calculated.
def calculate_accuracy(weights, dataset, correct_labels):
    mean_feature_of_dataset = np.array(np.mean(dataset, axis=1))
    std_feature_of_dataset = np.array(np.std(dataset, axis=1))

    y_final = probability_of_sample(weights, mean_feature_of_dataset, std_feature_of_dataset)
    count = 0
    accurate = 0
    correctly_classified_class_0_sample = 0
    correctly_classified_class_1_sample = 0
    for y in y_final:
        # print(y)
        if y < 0.5:
            if correct_labels[0, count] == 1:
                accurate += 1
                correctly_classified_class_1_sample += 1
        else:
            if correct_labels[0, count] == 0:
                accurate += 1
                correctly_classified_class_0_sample += 1
        count += 1

    print("Total number of Samples in the given classes : ", count)
    print("Accuracy of class 0 samples: ", (correctly_classified_class_0_sample / (correct_labels.size / 2)) * 100)
    print("Accuracy of class 1 samples: ", (correctly_classified_class_1_sample / (correct_labels.size / 2)) * 100)
    print("overall accuracy of the given dataset: ", (accurate / correct_labels.size) * 100)


weight = [10, 10, 10]
weight = assent(weight, weight, 0.005)

print("Passing training data to check accuracy")
calculate_accuracy(weight, Data.trX, Data.trY)

print("\n\nPassing testing data to check accuracy")
calculate_accuracy(weight, Data.tsX, Data.tsY)
