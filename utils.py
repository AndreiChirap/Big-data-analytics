import numpy as np
from matplotlib import pyplot as plt
import itertools


def read_data(file_path):
    path_elements = file_path.split("/")
    db_entries = path_elements[-1][5:-4]
    x = np.zeros([int(db_entries), 4])
    y = np.zeros(int(db_entries), dtype=int)
    k = 0
    with open(file_path, 'r') as fp:
        line = fp.readline()
        if line != "":
            line_values = line.split(",")
            aux_x = np.array(line_values[:-1]).astype(np.float)
            iris_type = determine_type(line_values[4])
            y[k] = iris_type
            x[k] = aux_x
            k += 1
            while line:
                line = fp.readline()
                if line != "":
                    line_values = line.split(",")
                    aux_x = np.array(line_values[:-1]).astype(np.float)
                    iris_type = determine_type(line_values[4])
                    y[k] = iris_type
                    x[k] = aux_x
                    k += 1
    iris = {}
    iris["data"] = x
    iris["target"] = y
    return iris


def determine_type(s):
    # 0 = setosa
    # 1 = versicolor
    # 2 = virginica
    if s.find("setosa") != -1:
        return 0
    elif s.find("versicolor") != -1:
        return 1
    elif s.find("virginica") != -1:
        return 2
    
    
def get_prediction(y_true, partition):
    y_true = y_true+1
    indices = np.argmax(partition, axis = 1)
    new_partition = np.zeros_like(partition)
    np.put_along_axis(new_partition, indices[..., None], 1, axis=1)
    categorical_partition = new_partition*y_true[..., None]
    centers_class_prob = { i: np.array([(((categorical_partition[:,i])[categorical_partition[:,i]!=0]) == j).sum()/((categorical_partition[:,i])[categorical_partition[:,i]!=0]).shape[0] for j in np.unique(y_true)]) for i in range(partition.shape[1])}
    
    center_class_map = {}
    for center, prob in centers_class_prob.items():
        center_class_map[center] = np.argmax(prob)

    y_pred = np.array([ center_class_map[np.argmax(categorical_partition[i])] for i in range(categorical_partition.shape[0])])
    return y_pred


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')