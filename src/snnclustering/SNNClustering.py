from sparse_similarity_computation import compute_knn
import numpy as np


class SNNClustering:
    UNCLASSIFIED = -1
    NOISE = -2
    CORE_POINT = 1
    NOT_CORE_POINT = 0

    num_points = None
    epsilon = None
    minpts = None
    Smat = None
    corepoints = []
    d_point_cluster_id = {}
    
   
    def __init__(self, epsilon, minpts, Smat):
        self.num_points = Smat.shape[0]
        self.epsilon = epsilon
        self.minpts = minpts
        self.Smat = Smat

        for i in range(self.num_points):
            self.d_point_cluster_id[i] = self.UNCLASSIFIED;
        
   
    def get_epsilon_neighbours(self, index):
        epsilon_neighbours_t = []
        for i in range(self.num_points):
            if i == index:
                continue
            if self.Smat[index, i] < self.epsilon:
                continue
            else:
                epsilon_neighbours_t.append(i)

        return epsilon_neighbours_t

    def fit_predict(self):        
        '''
        Performs the clustering and returns the assignments and the corepoints id's
        '''
        cluster_id=1
        for i in range(self.num_points):
            if self.d_point_cluster_id[i] == self.UNCLASSIFIED:
                if self.expand(i, cluster_id) == self.CORE_POINT:
                    self.corepoints.append(i)
                    cluster_id += 1
                    
        point_cluster_id = np.ones((self.num_points,)) * self.UNCLASSIFIED; #array
        for i in range(self.num_points):
            point_cluster_id[i] = self.d_point_cluster_id[i]
        
        return point_cluster_id, self.corepoints


    def expand(self, index, cluster_id):
        return_value = self.NOT_CORE_POINT
        seeds = self.get_epsilon_neighbours(index)
        
        num_epsilon_neighbors = len(seeds)
        if num_epsilon_neighbors < self.minpts:
            self.d_point_cluster_id[index] = self.NOISE #marked as noise
        else:
            self.d_point_cluster_id[index] = cluster_id
            for j in range(num_epsilon_neighbors):
                self.d_point_cluster_id[seeds[j]] = cluster_id
            j = 1
            while True:
                self.spread(seeds[j], seeds, cluster_id)
                if j == (len(seeds)-1):
                    break
                j += 1
            return_value = self.CORE_POINT
        return return_value

    def spread(self, index, seeds, cluster_id):
        spread_neighbors = self.get_epsilon_neighbours(index)
        num_spread_neighbors = len(spread_neighbors)

        if num_spread_neighbors >= self.minpts:
            for i in range(num_spread_neighbors):
                d = spread_neighbors[i]
                if self.d_point_cluster_id[d] == self.NOISE or self.d_point_cluster_id[d] == self.UNCLASSIFIED:
                    if self.d_point_cluster_id[d] == self.UNCLASSIFIED:
                        seeds.append(d)
                    self.d_point_cluster_id[d] = cluster_id
        return None


def shared_nn_sim(Smat, k):
    '''
    Smat: Similarity matrix computed previously with some SIMILARITY measure.
    k : Nr. of nearest neighbors to use.
    '''
    import numpy as np
    
    N = Smat.shape[0]
    nn = np.zeros(Smat.shape)
    k = np.min([k, N])
    
    for i in range(N):
        nn_ixs_i = np.argsort( Smat[i,:])
        nn_ixs_i = nn_ixs_i[-2:-(k+2):-1]
        nn[i, nn_ixs_i] = 1
        nn[i, i] = 0
    
    nn = nn.dot(nn.transpose())
    return nn
    


if __name__ == '__main__':
    import numpy as np
    from sklearn import cluster, datasets
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import euclidean_distances
    import pylab
    import sys
    from clustering_scores import clustering_scores

    #sys.path.append('/Volumes/SSDII/Users/juan/git/distributed-clustering/python/')

    noisy_circles = datasets.make_circles(n_samples=1000, factor=.1, noise=.05)
    DATA, y = noisy_circles
    DATA = StandardScaler().fit_transform(DATA)

    D = euclidean_distances(DATA, DATA)
    S = 1 - (D - np.min(D))/(np.max(D)-np.min(D))

    max_vm, max_vm_params = 0, [0,0,0]
    for K in range(40, 150,10):
        Snn = shared_nn_sim(S, K)
        for Eps in range(10,K,10):
            for MinPts in range(10,K,10):
                snnclust = SNNClustering(Eps, MinPts, Snn)
                assignments, corepoints = snnclust.fit_predict() 

                results = clustering_scores(y, assignments, False)

                if results['VM'] > max_vm:
                    max_vm = results['VM']
                    max_vm_params = {"epsilon":Eps, "minpts":MinPts, "K":K}

    # BEST CONFIGURATION
    print "Best VM score:%0.4f achieved with params epsilon:%d minpts:%d k:%d" % (max_vm, max_vm_params["epsilon"],max_vm_params["minpts"],max_vm_params["K"])

