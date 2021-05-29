from clustering import KMeansMultithreading, CMeansMultithreading, KMeans, CMeans
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt

def main():
    iris_dataset = load_iris()
    data = iris_dataset['data']
    
    kmeans = KMeans(k=3)
    kmeans.fit(data)

    cmeans = CMeans()
    cmeans.fit(data)

    kmeans_multithreading = KMeansMultithreading(k=3)
    kmeans_multithreading.fit(data)

    cmeans_multithreading = CMeansMultithreading()
    cmeans_multithreading.fit(data)
    
    
if __name__ == "__main__":
    main()


