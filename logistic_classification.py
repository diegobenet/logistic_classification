""" logistic_classification.py
    This python script implements the logistic classification with the functions provided by utilityfunction.py.

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
import time
import matplotlib.pyplot as plt

# import user-defined libraries
import utilityfunction as uf


def main():
    """"
    This main function calls all the functions needed to get the data and to implement the logistic regression
    with said data, as well as to build the confusion matrix and compute some performance metrics.

    INPUTS:
        none

    OUTPUTS:
        none
    """

    # load training and testing data
    # if flag = 1, the data is randomly generated using a seed, else the data is randomly generated from scratch
    # this is used so the the execution can be the same as the one used for the technical report
    flag = 1
    data_training, mean_training, std_training, \
        data_testing, mean_testing, std_testing = uf.load_training_data('heart.csv', flag)

    # visualize 10 random training samples
    # if flag = 1, the data is randomly generated using a seed, else the data is randomly generated from scratch
    flag = 1
    uf.visualize_random('Visualize 10 Random training samples', data_training, flag)

    # visualize 10 random testing samples
    # if flag = 1, the data is randomly generated using a seed, else the data is randomly generated from scratch
    flag = 1
    uf.visualize_random('Visualize 10 Random testing samples', data_testing, flag)

    # obtain the x values and the y values of the training data
    training_num_rows, num_cols = data_training.shape
    x_training = data_training[:, :-1]
    y_training = data_training[:, num_cols - 1].reshape(1, training_num_rows).T

    # obtain the x values and the y values of the testing data
    testing_num_rows = data_testing.shape[0]
    x_testing = data_testing[:, :-1]
    y_testing = data_testing[:, num_cols - 1].reshape(1, testing_num_rows).T

    # get training data normalized
    # if flag = 1, the data is randomly generated using a seed, else the data is randomly generated from scratch
    # this is used so the the execution can be the same as the one used for the technical report
    flag = 1
    x_training_scaled = uf.feature_scaling('Training data scaled', x_training, mean_training, std_training, flag)

    # initialize parameters
    stopping_criteria = 0.01
    learning_rate = 0.5

    # initialize w
    w = np.zeros_like(data_training[0, :]).reshape(1, num_cols).T

    # run the gradient descent method for parameter optimisation purposes
    w, cost, iteration = uf.gradient_descent_logistic(x_training_scaled, y_training, w, stopping_criteria,
                                                      learning_rate)
    # print parameters w
    uf.print_parameters(w)

    # get Testing data normalized
    # if flag = 1, the data is randomly generated using a seed, else the data is randomly generated from scratch
    # this is used so the the execution can be the same as the one used for the technical report
    flag = 1
    x_testing_scaled = uf.feature_scaling('Testing data scaled', x_testing, mean_testing, std_testing, flag)

    # predict testing data results
    testing_predicted = uf.compute_prediction_logistic(x_testing_scaled, w)

    # build and print the confusion matrix and its metrics
    uf.confusion_matrix(testing_predicted, y_testing)

    # print execution time and number of iterations taken
    print('\nRun time: ', time.process_time())
    print('Iterations: ', len(iteration))

    # display graph of the cost function
    uf.generate_cost_graph(cost, iteration)


main()
