import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
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


def multiclass_accuracy(prediction, ground_truth):
    accuracy = np.sum(prediction == ground_truth) / prediction.size

    return accuracy
