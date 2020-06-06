# pep8 (pycodestyle)
import numpy as np


def binary_classification_metrics(prediction: np.ndarray,
                                  ground_truth: np.ndarray) -> (float, float,
                                                                float, float):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    conf_mat = np.zeros((2, 2))
    for i in range(prediction.size):
        conf_mat[int(not prediction[i]),
                 int(not ground_truth[i])] += 1

    # precision calc
    if ((conf_mat[0, 0] == 0) and (conf_mat[0, 1] == 0)):
        precision = 0
    else:
        precision = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])

    # recall calc
    if ((conf_mat[0, 0] == 0) and (conf_mat[1, 0] == 0)):
        recall = 0
    else:
        recall = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[1, 0])

    # f1 calc
    if ((precision == 0) and (recall == 0)):
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # accuracy calc
    accuracy = (conf_mat[0, 0] + conf_mat[1, 1]) / np.sum(conf_mat)

    return (precision, recall, f1, accuracy)


def multiclass_accuracy(prediction: np.ndarray,
                        ground_truth: np.ndarray) -> float:
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    accuracy = prediction[prediction == ground_truth].size / prediction.size

    return accuracy
