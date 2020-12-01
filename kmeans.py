import dataset
from matplotlib.pyplot import axis
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg
from numpy.core.defchararray import array
from numpy.linalg.linalg import norm
from sklearn.cluster import KMeans as SK_KMeans
from sklearn.datasets import load_iris
from numpy.core.fromnumeric import shape
from time import time
from sklearn.metrics import confusion_matrix
import sys


class KMeans:
    def __init__(self, k = 4, max_iter = 15, tolerance = 0.00000001):
        self.k = k
        self.max_iter = max_iter
        self.tolerance = tolerance
        
    def get_centers(self):
        return self.__centers    
         
    def __init_centers(self):
        random_centers = np.random.randint(low = 0, high = self.__no_samples, size = (self.k))
        return self.__data[random_centers]
           
    def __update_w(self):
        self.__w[...] = 0
        distances = np.stack([np.linalg.norm(self.__data - self.__centers[i], axis = -1) for i in range(self.k)], axis = -1)
        index_array = np.argmin(distances, axis = -1)
        np.put_along_axis(self.__w, index_array[..., None], 1, axis=-1)
                    
    def __update_centroids(self):
        self.__previousCenters = self.__centers
        self.__centers = np.stack([ np.divide(np.sum(self.__data[self.__w[:,i]!=0], axis = 0), self.__w[:, i].sum()) for i in range(self.k) ], axis = 0)

    def __centroids_unchanged(self):
        differences = np.abs(np.subtract(self.__centers, self.__previousCenters))
        return (differences <= self.tolerance).sum() == np.prod(differences.shape)
    
    def fit(self, data):
        self.__data = data
        self.__no_samples = self.__data.shape[0]
        self.__w = np.zeros(shape = (self.__no_samples, self.k), dtype = np.uint8)
        self.__centers = self.__init_centers()

        t1 = time()
        for i in range(self.max_iter):
            self.__update_w()
            self.__update_centroids()
            if self.__centroids_unchanged():
                print("Algorithm stopped at iteration: %s" % i)
                break
        t2 = time()
        print(f"Kmeans time = {t2-t1}")
        
class KMeansNumba:
    pass


class CMeans:
    def __init__(self, C = 3, m = 2, max_iter = 15):
        self.C = C
        self.m = m
        self.max_iter = max_iter
          
    def __update_centroids(self):
        self.__centers = np.stack([np.divide(np.sum(np.multiply(self.__data, np.power(self.__w[:,i], self.m)[..., None]) ,axis = 0), np.sum(np.power(self.__w[:,i], self.m))) for i in range(self.C)], axis= 0)
    
    def __update_w(self):
        self.__w = np.divide(1, np.stack([np.array([np.power(np.divide(np.linalg.norm(self.__data - self.__centers[j], axis=-1), np.linalg.norm(self.__data - self.__centers[k], axis = -1)), 2/(self.m-1)) for k in range(self.C)]).sum(axis = 0) for j in range(self.C)], axis = -1))

    def fit(self, data):
        self.__data = data
        self.__no_samples = self.__data.shape[0]
        self.__features = self.__data.shape[1]
        self.__w = np.random.rand(self.__no_samples, self.C)
        self.__centers = np.empty(shape = (self.C, self.__features))
        
        t1 = time()
        for _ in range(self.max_iter):
            self.__update_centroids()
            self.__update_w()
        t2 = time()
        print(f"Cmeans time = {t2-t1}")
            
    def get_centers(self):
        return self.__centers
    
    def get_partition(self):
        return self.__w
    
class CMeansNumba:
    pass
                    
                
def create_fake_cluster(center, radius, no_points):    
    data = np.array(center)
    current_no_of_points = 0
    while current_no_of_points <= no_points:
        point = np.random.randint(low = min(center) - radius, high = max(center) + radius, size = (1,2))
        if (center[0] - point[0,0])**2 + (center[1] - point[0,1])**2 <= radius**2:
            data = np.vstack((data, point))
            current_no_of_points +=1
    return data

def main(*args, **kwargs):
    
    datasetFile = '1500'
    if sys.argv.count == 1:
        datasetFile = sys.argv[1]

    datasetPath = 'dataset/Iris-' + datasetFile + '.txt'
    print ("The script has the name %s" % (datasetPath))
    t1 = time()
    data = dataset.read_data(datasetPath)['data']
    t2 = time()
    print(f"Read db time = {t2-t1}")
    
    #my solution KMeans
    kmeans = KMeans(k = 3, max_iter=300)
    kmeans.fit(data)
    centers = kmeans.get_centers()
    
    #sklearn solution
    sk_kmeans = SK_KMeans(n_clusters = 3)
    #t3 = time()
    sk_kmeans.fit(data)
    #t4 = time()
    #print(f"Kmeans SK time = {t4-t3}")
    sk_centers = sk_kmeans.cluster_centers_

    #my solution CMeans
    #cmeans = CMeans(C = 3, m=2, max_iter=300)
    #cmeans.fit(data)
    #cmeans_centers = cmeans.get_centers()

      

    # #plot data + centers
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:,0], centers[:,1], color = "red")
    plt.scatter(sk_centers[:,0], sk_centers[:,1], color = "green")
    #plt.scatter(cmeans_centers[:,0], cmeans_centers[:,1], color = "yellow")
    plt.show()

if __name__ == "__main__":
    main()
    
    
    
    