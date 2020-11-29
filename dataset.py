import numpy as np


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
