
import dataset
import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans as SK_KMeans
from sklearn.datasets import load_iris
from numpy.core.fromnumeric import shape



class KMeans:
    def __init__(self, X, k = 4, max_iter = 300):
        #public
        self.k = k
        self.max_iter = max_iter
        
        #private
        self.__X = X
        self.__no_sampels = self.__X.shape[0]
        self.__w_ij = np.zeros(shape = (self.__no_sampels, self.k), dtype = np.uint8)
        self.__centers = self.__init_centers()
        
    #private    
    def __init_centers(self):
        random_centers = np.random.randint(low = 0, high = self.__no_sampels, size = (self.k))
        return self.__X[random_centers]
    
    #private    
    def __update_w_ij(self):
        for i in range(self.__no_sampels):
            min_dist = np.infty
            index = 0
            for j in range(self.k):
                dist = self.__norm_pow_2(self.__X[i], self.__centers[j])
                if min_dist > dist:
                    min_dist = dist
                    index = j
            self.__w_ij[i,:] = 0
            self.__w_ij[i,index] = 1
            
    #private
    def __update_centers(self):
        for i in range(self.k):
            sum_k = self.__w_ij[:,i].sum()
            self.__centers[i] = np.divide(np.sum(self.__X[self.__w_ij[:,i]!=0], axis = 0), sum_k)
    #public
    def run(self):
        for _ in range(self.max_iter):
            self.__update_w_ij()
            self.__update_centers()
    
    #public
    def get_centers(self):
        return self.__centers
    
    #private static
    @staticmethod
    def __norm_pow_2(v1, v2):
        return np.linalg.norm(v1 - v2)**2


def create_fake_cluster(center, radius, no_points):    
    data = np.array(center)
    current_no_of_points = 0
    while current_no_of_points <= no_points:
        point = np.random.randint(low = min(center) - radius, high = max(center) + radius, size = (1,2))
        if (center[0] - point[0,0])**2 + (center[1] - point[0,1])**2 <= radius**2:
            data = np.vstack((data, point))
            current_no_of_points +=1
    return data
        
def print_stop_time(start_time):
    print("--- Execution: %.2f seconds ---" % (round(time.time() - start_time, 2)))

def main(*args, **kwargs):
    # c1 = create_fake_cluster((100,100), 10, 40)
    # c2 = create_fake_cluster((-100, 100), 10, 40)
    # c3 = create_fake_cluster((-100, -100), 10, 40)
    # c4 = create_fake_cluster((100, -100), 10, 40)
    # data = np.vstack((c1, c2, c3, c4))
    
    #iris_data = load_iris()
    #data = iris_data.data

    data = dataset.read_data('dataset/Iris-1500.txt')['data']

    #my solution
    kmeans_start_time = time.time()

    kmeans = KMeans(data, k = 3)
    kmeans.run()

    print_stop_time(kmeans_start_time)
    centers = kmeans.get_centers()
    
    #sklearn solution
    sk_kmeans_start_time = time.time()

    sk_kmeans = SK_KMeans(n_clusters = 3, init = "random")
    sk_kmeans.fit(data)

    print_stop_time(sk_kmeans_start_time)
    sk_centers = sk_kmeans.cluster_centers_

    #plot data + centers
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:,0], centers[:,1], color = "red")
    plt.scatter(sk_centers[:,0], sk_centers[:,1], color = "green")
    plt.show()

if __name__ == "__main__":
    main()
    
    
    