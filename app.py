from time import time
from matplotlib import pyplot as plt
from clustering import KMeansMultithreading, CMeansMultithreading, KMeans, CMeans
from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix, determine_type, read_data, get_prediction

def main():
    
    t1 = time()
    iris = read_data(r"dataset/Iris-150.txt")
    t2 = time()
    print("\n")
    print(f"Read db time = {t2-t1}")
    
    data = iris["data"]
    y_true = iris["target"]
    
    print("\n")
    if True:
        kmeans_parallel = KMeansMultithreading(k=3, max_iter=100, tolerance= 0.0001)
        kmeans_parallel.fit(data)
        kmeans_parallel_centers = kmeans_parallel.get_centers()
        kmeans_parallel_partition = kmeans_parallel.get_partition()

        y_pred = get_prediction(y_true, kmeans_parallel_partition)
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, classes=["setosa","versicolor","virginica"])
        plt.show()
    print("\n")
    print("##############################################################################")
    print("\n")
    if True:
        cmeans_parallel = CMeansMultithreading(C=3, m=2, max_iter=100, tolerance=0.0001)
        cmeans_parallel.fit(data)
        cmeans_parallel_centers = cmeans_parallel.get_centers()
        cmeans_parallel_partition = cmeans_parallel.get_partition()
        
        y_pred = get_prediction(y_true, cmeans_parallel_partition)
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, classes=["setosa","versicolor","virginica"])
        plt.show()
    print("\n")
    print("##############################################################################")
    print("\n")
    if  True:
        kmeans_seq = KMeans(k=3, max_iter=100, tolerance= 0.0001)
        kmeans_seq.fit(data)
        kmeans_seq_centers = kmeans_seq.get_centers()
        kmeans_seq_partition = kmeans_seq.get_partition()

        y_pred = get_prediction(y_true, kmeans_seq_partition)
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, classes=["setosa","versicolor","virginica"])
        plt.show()    
    print("##############################################################################")
    print("\n")
    if True:
        cmeans_seq = CMeans(C=3, m=2, max_iter=100, tolerance=0.0001)
        cmeans_seq.fit(data)
        cmeans_seq_centers = cmeans_seq.get_centers()
        cmeans_seq_partition = cmeans_seq.get_partition()
        
        y_pred = get_prediction(y_true, cmeans_seq_partition)
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, classes=["setosa","versicolor","virginica"])
        plt.show()
    
    
if __name__ == "__main__":
    main()


