""" utilityfunction.py
    This python script has the functions needed to implement the logistic classification.

    Authors:
        Daniel Alberto Martínez Sánchez     534032
        Diego Elizondo Benet                567003
        Alejandro Flores Ramones            537489
        Karla Lira Rangel                   526389
    Emails:
        daniel.martinezs@udem.edu
        diego.elizondob@udem.edu
        alejandro.floresr@udem.edu
        karla.lira@udem.edu

    Institution: Universidad de Monterrey
    First created: Wednesday  11 Nov 2020

    We hereby declare that we've worked on this activity with academic integrity.
"""

# import standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def load_training_data(path_and_filename, flag):
    """"
    This function reads the data of an external file and uses it to initialize the training and testing data, for it to
    calculate some statistics like mean, media, std, min and max. At last it prints the results and converts it to a
    numpy-type matrix

    INPUTS:
        path_and_filename: String representing the name and location of the file.
        flag: Number representing if the randomized data should be seeded or not.

    OUTPUTS:
        data_training: numpy-type matrix with the attributes representing each feature of training.
        mean_training: numpy-type vector with the mean of each feature of training.
        std_training: numpy-type vector with the standard deviation of training.
        data_testing: numpy-type matrix with the attributes representing each feature of testing.
        mean_testing: numpy-type vector with the mean of each feature of testing.
        std_testing: numpy-type vector with the standard deviation of testing.
    """

    data_training = []
    mean_training = []
    std_training = []
    min_training = []
    max_training = []
    median_training = []

    data_testing = []
    mean_testing = []
    std_testing = []
    min_testing = []
    max_testing = []
    median_testing = []

    try:
        # read the file
        data = pd.read_csv(path_and_filename)

        # if flag is 1 use the seed '55' (used for the technical report)
        if flag == 1:
            seed = 55
            np.random.seed(seed)

        # shuffle data
        data = data.iloc[np.random.permutation(len(data))]

        # use 80% for training and 20% for testing
        data_training, data_testing = np.split(data, [int(.8 * len(data))])

        # compute statistics
        mean_training = np.mean(data_training.to_numpy()[:, :-1], axis=0)
        std_training = np.std(data_training.to_numpy()[:, :-1], axis=0)
        min_training = np.min(data_training.to_numpy()[:, :-1], axis=0)
        max_training = np.max(data_training.to_numpy()[:, :-1], axis=0)
        median_training = np.median(data_training.to_numpy()[:, :-1], axis=0)

        mean_testing = np.mean(data_testing.to_numpy()[:, :-1], axis=0)
        std_testing = np.std(data_testing.to_numpy()[:, :-1], axis=0)
        min_testing = np.min(data_testing.to_numpy()[:, :-1], axis=0)
        max_testing = np.max(data_testing.to_numpy()[:, :-1], axis=0)
        median_testing = np.median(data_testing.to_numpy()[:, :-1], axis=0)

    except IOError as e:
        print(e)
        exit(1)

    # print statistics, mean, max, min, deviation and testings data and training data
    print('-' * 90)
    print('Training data and Y (target) outputs')
    print('-' * 90)
    print(data_training)
    print('-' * 90)
    print('Training Statistics')
    print('-' * 90)
    print('Training mean:\n', mean_training)
    print('\nTraining standard deviation:\n', std_training)
    print('\nTraining max:\n', max_training)
    print('\nTraining min:\n', min_training)
    print('\nTraining median:\n', median_training)
    print('-' * 90)
    print('Testing data and Y (target) outputs')
    print('-' * 90)
    print(data_testing)
    print('-' * 90)
    print('Testing Statistics')
    print('-' * 90)
    print('Testing mean:\n', mean_testing)
    print('\nTesting standard deviation:\n', std_testing)
    print('\nTraining max:\n', max_testing)
    print('\nTraining min:\n', min_testing)
    print('\nTraining median:\n', median_testing)

    return data_training.to_numpy(), mean_training, std_training, data_testing.to_numpy(), mean_testing, std_testing


def visualize_random(title, data, flag):
    """"
    This function prints 10 randomly selected samples from the data-set on the command-line.

    INPUTS:
        title: string representing the header for visual purposes.
        data: numpy-type matrix with the attributes representing each feature.
        flag: Number representing if the randomized data should be seeded or not.

    OUTPUTS:
        none
    """
    print('-' * 90)
    print(title)
    print('-' * 90)
    num_rows = data.shape[0]

    if flag == 1:
        seed = 23
        np.random.seed(seed)
    rand = np.random.random_integers(0, num_rows, 10)

    np.set_printoptions(suppress=True)
    for i in range(len(rand)):
        print(data[rand[i], :])


def feature_scaling(title, x, mean, std, flag):
    """"
    This function implements feature scaling on the attributes of each feature of the data-set provided and prints it
    on the command line.

    INPUTS:
        title: string representing the header for visual purposes.
        x: numpy-type matrix with the attributes representing each feature.
        mean: numpy-type vector with the mean of each feature.
        std: numpy-type vector with the standard deviation.
        flag: Number representing if the randomized data should be seeded or not.

    OUTPUTS:
        x_scaled: numpy-type matrix with the attributes representing the scaled data.
    """

    num_cols = x.shape[1]
    x_scaled = np.zeros_like(x)

    # scale each feature
    for i in range(num_cols):
        scale = (x[:, i] - mean[i]) / (math.sqrt((std[i] ** 2) + (10 ** -8)))
        x_scaled[:, i] = scale
    np.set_printoptions(precision=3)
    print('-' * 90)
    print(title)
    print('-' * 90)

    num_rows = x_scaled.shape[0]
    if flag == 1:
        seed = 34
        np.random.seed(seed)
    rand = np.random.random_integers(0, num_rows, 10)

    for i in range(len(rand)):
        print(x_scaled[rand[i], :])
    return x_scaled


def gradient_descent_logistic(x_training, y_training, w, stopping_criteria, learning_rate):
    """"
    This function calls the function compute_gradient_of_cost_function_logistic and compute_l2_norm_logistic
    to implement the gradient descent for the logistic regression through a while loop.
    Additionally, it calls compute_cost_function_logistic to build a list with the cost through iterations.

    INPUTS:
        x_training: numpy-type matrix.
        y_training: numpy-type vector.
        w: numpy-type vector.
        stopping_criteria: number representing when to stop the logistic regression.
        learning_rate: number representing the learning rate used in the logistic regression.

    OUTPUTS:
        w: numpy-type vector representing the parameters obtained.
        cost: numpy-type vector representing the cost obtained of each iterations of the logistic regression.
        iteration: numpy-type vector with number representing the times the loop had to iterate.
    """

    num_rows, num_cols = x_training.shape

    x0 = [1] * num_rows

    x = np.c_[x0, x_training]
    cost = []
    iteration = []
    l2_norm = 100
    i = 0

    while l2_norm > stopping_criteria:
        # compute gradient of cost function
        gradient_cost_function = compute_gradient_of_cost_function_logistic(x, y_training, w)

        # compute the cost function (to plot)
        cost.append(compute_cost_function_logistic(x, y_training, w))

        # update parameters
        w = w - learning_rate * gradient_cost_function

        # compute the L2 norm
        l2_norm = compute_l2_norm_logistic(gradient_cost_function)
        iteration.append(i)
        i += 1

    return w, cost, iteration


def eval_hypothesis_function_logistic(w, x):
    """
    This function evaluates the hypothesis function needed for the logistic regression.

    INPUTS:
        w: numpy-type vector with the results of the obtained parameters on the logistic regression.
        x_training_data: numpy-type matrix.

    OUTPUTS
        hypothesis_function: numpy-type vector.
    """

    wx = np.matmul(x, w)
    hypothesis_function = 1 / (1 + np.exp(-wx))
    return hypothesis_function


def compute_gradient_of_cost_function_logistic(x, y, w):
    """
    This function implements the gradient of the cost function used for the logistic regression.
    It also calls the eval_hypothesis_function_multivariate function to then calculate the residual with its value.

    INPUTS:
        x: numpy-type matrix with the attributes representing each feature.
        y: numpy-type vector.
        w: numpy-type vector with the results of the obtained parameters on the logistic regression.

    OUTPUTS
        grad_cost_function: numpy-type vector
    """

    n = x.shape[0]

    hypothesis_function = eval_hypothesis_function_logistic(w, x)
    residual = hypothesis_function - y

    grad_cost_function = np.matmul(residual.T, x).T / n

    return grad_cost_function


def compute_cost_function_logistic(x, y, w):
    """
    This function implements the  cost needed for the logistic regression.
    It also calls the eval_hypothesis_function_multivariate function to then calculate the residual with its value.

    INPUTS:
        x: numpy-type matrix with the attributes representing each feature.
        y: numpy-type vector.
        w: numpy-type vector with the results of the obtained parameters on the logistic regression.

    OUTPUTS
        cost: numpy-type vector representing the cost of each parameter obtained.
    """

    hypothesis_function = eval_hypothesis_function_logistic(w, x)
    cost = -(y * np.log(hypothesis_function) + (1 - y) * np.log(1 - hypothesis_function)).sum()
    return cost


def compute_l2_norm_logistic(gradient_cost_function):
    """
    This function gets the l2 norm needed for the logistic regression.

    INPUTS:
        grad_cost_function: numpy-type vector

    OUTPUTS
        l2_norm: Number representing the l2 / euclidean norm
    """

    l2_norm = np.sqrt(np.matmul(gradient_cost_function.T, gradient_cost_function).sum())
    return l2_norm


def print_parameters(w):
    """
    This function print the parameters 'w' obtained in the logistic regression.

    INPUTS:
        w: numpy-type vector

    OUTPUTS
        none
    """

    print('-' * 90)
    print('W Parameters')
    print('-' * 90)
    print(w)


def compute_prediction_logistic(x_testing, w):
    """
    This function uses the hypothesis function with the obtained w parameters and the testing data-set to predict
    the result given with 0 or 1.

    INPUTS:
        x_testing: numpy-type matrix
        w: numpy-type vector

    OUTPUTS
        predicted_classes: numpy-type vector
    """

    num_rows = x_testing.shape[0]

    x0 = [1] * num_rows

    x = np.c_[x0, x_testing]

    predicted_classes = eval_hypothesis_function_logistic(w, x)

    # if the hypothesis function for a sample is at least 0.5, consider it a positive class,
    # otherwise, a negative class
    for i in range(num_rows):
        if predicted_classes[i] >= 0.5:
            predicted_classes[i] = 1
        else:
            predicted_classes[i] = 0
        i += 1

    return predicted_classes


def confusion_matrix(predicted, y):
    """
    This function computes the confusion matrix using the predictions previously obtained.
    It then calculates some metrics like accuracy, precision, recall, sensitivity and F1-score.
    At last it prints the confusion matrix and the obtained metrics.

    INPUTS:
        predicted: numpy-type vector
        y: numpy-type vector

    OUTPUTS
        none
    """

    print('-' * 90)
    print('Confusion Matrix')
    print('-' * 90)
    num_rows = predicted.shape[0]

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # build the confusion matrix
    for i in range(num_rows):
        if predicted[i] == 1 and y[i] == 1:
            true_positives += 1
        elif predicted[i] == 1 and y[i] == 0:
            false_positives += 1
        elif predicted[i] == 0 and y[i] == 0:
            true_negatives += 1
        elif predicted[i] == 0 and y[i] == 1:
            false_negatives += 1

    total = true_positives + false_positives + true_negatives + false_negatives

    # compute metrics from the confusion matrix
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    # print the confusion matrix
    space = '\t\t\t\t'
    print(space, '                       Actual class')
    print(space, '                      1           0')
    print(space, '                 ┌----------┬----------┐')
    print(space, '               1 |    TP    |    FP    |')
    print(space, '   Predicted     |   ', true_positives, '\t |   ', false_positives, '  \t|   ')
    print(space, '   class         ├----------┼----------┤')
    print(space, '               0 |    FN    |    TN    |')
    print(space, '                 |   ', false_negatives, '\t |   ', true_negatives, '\t|   ')
    print(space, '                 └----------┴----------┘')
    print('Legend: TP = True Positive\tFP = False Positive\tFN = False Negative\tTN = True Negative')

    # print the confusion matrix metrics
    print('\nMetrics:')
    print('Accuracy: \t\t', round(accuracy, 2))
    print('Precision: \t\t', round(precision, 2))
    print('Recall: \t\t', round(recall, 2))
    print('Specificity: \t', round(specificity, 2))
    print('F1 Score: \t\t', round(f1_score, 2))

    print('\nTotal data: ', total)


def generate_cost_graph(cost, iteration):
    """
    This function plots and graphs the cost obtained on the linear regressions and compares it to the number of
    iteration it finds itself in.

    INPUTS:
        cost: numpy-type vector representing the cost obtained of each iterations of the logistic regression.
        iteration: numpy-type vector with number representing the times the loop had to iterate.

    OUTPUTS
        none
    """

    f, ax = plt.subplots(1)
    plt.figure(1)
    plt.plot(iteration, cost, color="green")
    plt.legend(["Cost", "regression line"])
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid("True")
    ax.set_ylim(ymin=0)
    plt.show()
