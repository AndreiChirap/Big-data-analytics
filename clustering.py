from numba import njit, prange
from time import time
import numpy as np


@njit(parallel=True, fastmath=True)
def fast_norm(vec):
    return np.array([np.sqrt(np.sum(np.power(vec[i,:], 2))) for i in prange(vec.shape[0])])


class KMeansMultithreading:
    def __init__(self, k=4, max_iter=15, tolerance=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tolerance = tolerance
        
    def get_centers(self):
        return self.__centers
        
    def get_partition(self):
        return self.__w
    
    def __init_centers(self):
        random_centers = np.random.randint(low=0, high=self.__no_samples, size=self.k)
        return self.__data[random_centers]
    
    def __update_w(self):
        self.__w[...] = 0
        distances = np.stack([j for j in KMeansMultithreading.__fast_distances(self.k, self.__data, self.__centers)], axis=-1)
        np.put_along_axis(self.__w, np.argmin(distances, axis=-1)[..., None], 1, axis=-1)
        
    def __centroids_unchanged(self):
        differences = np.abs(np.subtract(self.__centers, self.__previous_centers))
        return (differences <= self.tolerance).sum() == np.prod(differences.shape)
                                   
    def fit(self, data):
        self.__data = data
        self.__no_samples = self.__data.shape[0]
        self.__w = np.zeros(shape = (self.__no_samples, self.k), dtype = np.uint8)
        self.__centers = self.__init_centers()
        
        t1 = time()
        for i in range(self.max_iter):
            self.__update_w()
            self.__previous_centers = self.__centers.copy()
            KMeansMultithreading.__fast_update_centroids(self.k, self.__w, self.__data, self.__centers)
            if self.__centroids_unchanged():
                print(f"Algorithm stopped at iteration: {i}")
                break
        t2 = time()
        print(f"KMeans(multithreading) time = {t2-t1}")

    @staticmethod
    @njit(parallel = True, fastmath=True)
    def __fast_distances(k, data, centers):
        return [(fast_norm(data - centers[i])) for i in prange(k)]
            
    @staticmethod
    @njit(parallel = True, fastmath=True)    
    def __fast_update_centroids(k, w, data, centers):
        for i in prange(k):
            centers[i] = np.divide(np.sum(data[w[:,i] !=0 ], axis=0), w[:,i].sum())
    

class CMeansMultithreading:
    def __init__(self, C=3, m=2, max_iter=15, tolerance=1e-4):
        self.C = C
        self.m = m
        self.max_iter = max_iter
        self.tolerance = tolerance
        
    def get_centers(self):
        return self.__centers
    
    def get_partition(self):
        return self.__w         
    
    @staticmethod  
    @njit(parallel=True, fastmath=True)
    def __fast_update_centroids(c, m, data, w, centers):
        for k in prange(c):
            num = np.zeros_like(centers[k])
            denom = 0.0
            for x in prange(data.shape[0]):
                num += data[x]*(w[x,k])**m
                denom += (w[x,k])**m
            centers[k] = num/denom
        
    @staticmethod    
    @njit(parallel=True, fastmath=True)        
    def __fast_update_w(c, m, data, w, centers):
        for i in prange(data.shape[0]):
            for j in prange(c):
                s = 0.
                num = np.linalg.norm(data[i] - centers[j])
                for k in prange(c):
                    denom = np.linalg.norm(data[i] - centers[k])
                    fraction = num/denom
                    s+=(fraction)**(2/(m-1))
                w[i,j] = 1/s
    
    def __centroids_unchanged(self):
        differences = np.abs(np.subtract(self.__centers, self.__previous_centers))
        return (differences <= self.tolerance).sum() == np.prod(differences.shape)
               
    def fit(self, data):
        self.__data = data
        self.__no_samples = self.__data.shape[0]
        self.__features = self.__data.shape[1]
        self.__w = np.random.rand(self.__no_samples, self.C)
        self.__centers = np.empty(shape = (self.C, self.__features))
        
        t1 = time()
        for i in range(self.max_iter):
            self.__previous_centers = self.__centers.copy()
            CMeansMultithreading.__fast_update_centroids(self.C, self.m, self.__data, self.__w, self.__centers)
            CMeansMultithreading.__fast_update_w(self.C, self.m, self.__data, self.__w, self.__centers)
            if self.__centroids_unchanged():
                print(f"Algorithm stopped at iteration: {i}")
                break
        t2 = time()
        print(f"CMeans(multithreading) time = {t2-t1}")

            
class KMeans:
    def __init__(self, k=4, max_iter=15, tolerance=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tolerance = tolerance
        
    def get_centers(self):
        return self.__centers
    
    def get_partition(self):
        return self.__w         
        
    def __init_centers(self):
        random_centers = np.random.randint(low=0, high=self.__no_samples, size=self.k)
        return self.__data[random_centers]
           
    def __update_w(self):
        self.__w[...] = 0
        distances = np.stack([np.linalg.norm(self.__data - self.__centers[i], axis=-1) for i in range(self.k)], axis=-1)
        np.put_along_axis(self.__w, np.argmin(distances, axis = -1)[..., None], 1, axis=-1)
                    
    def __update_centroids(self):
        self.__centers = np.stack([ np.divide(np.sum(self.__data[self.__w[:,i] !=0 ], axis=0), self.__w[:, i].sum()) for i in range(self.k) ], axis=0)
    
    def __centroids_unchanged(self):
        differences = np.abs(np.subtract(self.__centers, self.__previous_centers))
        return (differences <= self.tolerance).sum() == np.prod(differences.shape)
    
    def fit(self, data):
        self.__data = data
        self.__no_samples = self.__data.shape[0]
        self.__w = np.zeros(shape = (self.__no_samples, self.k), dtype=np.uint8)
        self.__centers = self.__init_centers()

        t1 = time()
        for i in range(self.max_iter):
            self.__update_w()
            self.__previous_centers = self.__centers
            self.__update_centroids()
            if self.__centroids_unchanged():
                print(f"Algorithm stopped at iteration: {i}")
                break
        t2 = time()
        print(f"KMeans(iterative) time = {t2-t1}")


class CMeans:
    def __init__(self, C = 3, m = 2, max_iter = 15, tolerance = 1e-4):
        self.C = C
        self.m = m
        self.max_iter = max_iter
        self.tolerance = tolerance
        
    def get_centers(self):
            return self.__centers
    
    def get_partition(self):
        return self.__w  
    
    def __update_centroids(self):
        self.__centers = np.stack([np.divide(np.sum(np.multiply(self.__data, np.power(self.__w[:,i], self.m)[..., None]) ,axis = 0), np.sum(np.power(self.__w[:,i], self.m))) for i in range(self.C)], axis= 0)
    
    def __update_w(self):
        self.__w = np.divide(1, np.stack([np.array([np.power(np.divide(np.linalg.norm(self.__data - self.__centers[j], axis=-1), np.linalg.norm(self.__data - self.__centers[k], axis = -1)), 2/(self.m-1)) for k in range(self.C)]).sum(axis = 0) for j in range(self.C)], axis = -1))

    def __centroids_unchanged(self):
        differences = np.abs(np.subtract(self.__centers, self.__previous_centers))
        return (differences <= self.tolerance).sum() == np.prod(differences.shape)
    
    def fit(self, data):
        self.__data = data
        self.__no_samples = self.__data.shape[0]
        self.__features = self.__data.shape[1]
        self.__w = np.random.rand(self.__no_samples, self.C)
        self.__centers = np.empty(shape = (self.C, self.__features))
        
        t1 = time()
        for i in range(self.max_iter):
            self.__previous_centers = self.__centers.copy()
            self.__update_centroids()
            self.__update_w()
            if self.__centroids_unchanged():
                print(f"Algorithm stopped at iteration: {i}")
                break
            
        t2 = time()
        print(f"CMeans(iterative) time = {t2-t1}")