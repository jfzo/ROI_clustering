import numpy as np
import matplotlib.pyplot as plt
import dcm2 as dcm
import logging

def write_results(s):
    logging.info(s)

##
def snn_experiments(DATA, true_labels):
    write_results("SNN Experiments")

    import sys
    sys.path.append("snnclustering")
    from sklearn.metrics.pairwise import euclidean_distances
    import numpy as np
    import pylab
    from clustering_scores import clustering_scores
    from SNNClustering import SNNClustering, shared_nn_sim
    from sparse_similarity_computation import compute_knn

    D = euclidean_distances(DATA, DATA)
    S = 1 - (D - np.min(D))/(np.max(D)-np.min(D)) # transforming to a sim mat.

    write_results("Starting parameter tuning procedure...")
    for K in [50, 100, 150, 200]:
        Snn = shared_nn_sim(S, K)

        max_vm, max_sil, max_vm_params = 0, 0, [0, 0, K]
        for Eps in range(10,K, 10):
            for MinPts in range(10,K,10):
                snnclust = SNNClustering(Eps, MinPts, Snn)
                assignments, corepoints = snnclust.fit_predict() 
                #results = clustering_scores(y, assignments, False)
                
                if len(np.unique(assignments)) > 1:
                    current_vm = metrics.v_measure_score(true_labels, assignments)
                    if current_vm > max_vm:
                        max_vm = current_vm
                        max_sil = metrics.silhouette_score(DATA, assignments, metric='euclidean')
                        max_vm_params = Eps, MinPts, K
                #else:
                #    logging.warn("Uups!  No se pudo encontrar mas de un cluster.")

                    #write_results("Eps: {0} MinPts: {1} K:{2} -- VM: {3} ({4})".format(Eps, MinPts, K, results["VM"], max_vm) )


    # BEST CONFIGURATION
    write_results("Best VM({:.3f}) Silhouette({:.3f}) is achieved with Eps: {:d} MinPts: {:d} K: {:d}".format(max_vm, max_sil, max_vm_params[0], max_vm_params[1], max_vm_params[2]) )
    '''
    write_results("Homogeneity: {0:.3f}".format(metrics.homogeneity_score(true_labels, CL)) )
    write_results("Completeness: {0:.3f}".format(metrics.completeness_score(true_labels, CL)) )
    write_results("V-measure: {0:.3f}".format(metrics.v_measure_score(true_labels, CL)) )
    write_results("Silhouette: {0:.3f}".format(metrics.silhouette_score(DATA, CL, metric='euclidean')) )

    '''
    #knn_info = compute_knn(S, max_vm_params[2])  # sparsify the similarity matrix
    #snn_sim = compute_snn(knn_info)  # obtains the snn similarity matrix
    #CP, NCP, NP, CL = snn_clustering(snn_sim, max_vm_params[0], max_vm_params[1])
    #write_results("#core_points:", len(CP), "#non_core_points:", len(NCP), "#noisy_points:", len(NP), "#Clusters:", len(np.unique(CL[CP]) ))


def kmeans_experiments(DATA, true_labels, k):
    write_results("Kmeans Experiments")

    from sklearn.cluster import KMeans
    from sklearn.model_selection import GridSearchCV
    from sklearn import metrics
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics.pairwise import pairwise_distances

    param_grid = [
        {'n_clusters': range(3, 10)}
    ]
    km = GridSearchCV(estimator=KMeans(init='k-means++'), param_grid=param_grid, cv=3, scoring='f1_weighted')

    kmeans_model = km.fit(DATA, true_labels).best_estimator_
    labels = kmeans_model.labels_

    write_results("Best VM({:.3f}) Silhouette({:.3f}) is achieved with K: {:d}".format(
        metrics.v_measure_score(true_labels, labels), metrics.silhouette_score(DATA, labels, metric='euclidean'),
        kmeans_model.get_params()['n_clusters']) )

    #write_results("Homogeneity: {0:.3f}".format(metrics.homogeneity_score(true_labels, km.labels_)) )
    #write_results("Completeness: {0:.3f}".format(metrics.completeness_score(true_labels, km.labels_)) )
    #write_results("V-measure: {0:.3f}".format(metrics.v_measure_score(true_labels, km.labels_)) )
    #write_results("Silhouette: {0:.3f}".format(metrics.silhouette_score(DATA, labels, metric='euclidean')) )
    '''
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html#sklearn.metrics.pairwise.pairwise_distances
    '''
    # Antes de aplicar, generar un grafo de vecinos cercanos y entregarle
    # su matrix de adyacencia a la funcion spectral_clustering .-
    #DM = pairwise_distances(newYmat[:,0:-1], newYmat[:,0:-1])

    #SC = SpectralClustering(n_clusters=6, affinity='nearest_neighbors', n_neighbors=50, eigen_solver='arpack')
    #SC.fit(DATA)

def snn_spectral_experiments(DATA, true_labels, k):
    write_results("Spectral Experiments")
    from sklearn import metrics
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.cluster import spectral_clustering
    import sys
    sys.path.append("snnclustering")
    from sklearn.metrics.pairwise import euclidean_distances
    import numpy as np
    import pylab
    from clustering_scores import clustering_scores
    from SNNClustering import snn_clustering, compute_snn
    from sparse_similarity_computation import compute_knn

    D = euclidean_distances(DATA, DATA)
    S = np.exp(-D / D.std())
    #labels = spectral_clustering(similarity, n_clusters=k, eigen_solver='arpack')

    for K in [50, 70, 90, 110, 130]:
        knn_info = compute_knn(S, K) # sparsify the similarity matrix
        snn_sim = compute_snn(knn_info) # obtains the snn similarity matrix

        labels = spectral_clustering(snn_sim, n_clusters=k, eigen_solver='arpack')
        results = clustering_scores(true_labels, labels, display=False)  # dict containing {'E','P','ARI','AMI','NMI','H','C','VM'}
        write_results("K: {0} -- VM: {1}".format(K, results["VM"]) )#, "(", max_vm, ")"

        #print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels, labels))
        #print("Completeness: %0.3f" % metrics.completeness_score(true_labels, labels))
        #print("V-measure: %0.3f" % metrics.v_measure_score(true_labels, labels))

    '''
    Basado en el ejemplo disponible en:
    http://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html#sphx-glr-auto-examples-cluster-plot-segmentation-toy-py
    '''

def spectral_experiments(DATA, true_labels, k):
    write_results("Spectral Experiments")
    from sklearn import metrics
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.cluster import spectral_clustering
    import sys
    sys.path.append("snnclustering")
    from sklearn.metrics.pairwise import euclidean_distances
    import numpy as np
    import pylab
    from clustering_scores import clustering_scores
    from SNNClustering import snn_clustering, compute_snn
    from sparse_similarity_computation import compute_knn

    D = euclidean_distances(DATA, DATA)
    S = np.exp(-D / D.std())
    #labels = spectral_clustering(similarity, n_clusters=k, eigen_solver='arpack')

    labels = spectral_clustering(S, n_clusters=k, eigen_solver='arpack')
    results = clustering_scores(true_labels, labels, display=False)  # dict containing {'E','P','ARI','AMI','NMI','H','C','VM'}
    write_results("V-measure: {0:.3f}".format(results["VM"]) )#, "(", max_vm, ")"
    write_results("Silhouette: {0:.3f}".format(metrics.silhouette_score(DATA, labels, metric='euclidean')) )


if __name__ == '__main__':
    import sys
    from sklearn import metrics
#
#    if len(sys.argv) < 2:
#        logging.error("Se debe agregar el valor del parametro SNR.")
#        sys.exit()
#
    logging.basicConfig(filename='dcm_snr-snn-nnodos30.log', level=logging.INFO)

    
    #for nnodos in [5,10,30]:
    for nnodos in [30]:
        for nsignals in [100, 500, 1000]:
            for desb in [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:
            	for snr in [0.2, 0.5, 1.0, 2.0, 4.0]:
                    N_cluster = {'background':5000}
                    for i in range(1,nnodos+1):
                        ndstr = 'n'+str(i)
                        N_cluster[ndstr] = nsignals
                    N_cluster['n1'] = round(nsignals * desb)
                    
                    Y_data, U_data, H = dcm.get_dcm_data(SNR=snr, N_nds=nnodos, N_cluster=N_cluster)
    
                    #Y_mat = np.hstack(Y_data.values())
                    #Y_mat = Y_mat.transpose()
                    #Y_shuffled = np.copy(Y_mat)
                    #np.random.shuffle(Y_shuffled)
                    ldata = []
                    class_cnt = 1
                    for label in Y_data.keys():
                        ldata.append(np.vstack([Y_data[label], np.zeros((Y_data[label].shape[1],))+class_cnt]) )
                        class_cnt += 1
                    DATA = np.hstack(ldata)
                    DATA = DATA.transpose()

                    true_labels = DATA[:,-1]
                    DATA = DATA[:,0:-1]

                    write_results(
                        "Configuracion nnodos:{:d} nsenales:{:d} pct.desbalance:{:.4f} snr:{:.4f}".format(nnodos, nsignals, desb, snr))
                    #write_results("\nKMEANS Clustering")
                    #kmeans_experiments(DATA, true_labels, 6)
                    write_results("\nSNN Clustering")
                    snn_experiments(DATA, true_labels)

                    
                    
                    
                    
#
#                        write_results("Configuracion nnodos:%d nsenales:%d snr:%f pct.desbalance:%d".format(nnodos, nsignals, snr, desb))
#                        ##
#                        ldata = []
#                        class_cnt = 1
#                        for label in Y_data.keys():
#                            ldata.append(np.vstack([Y_data[label], np.zeros((Y_data[label].shape[1],))+class_cnt]) )
#                            class_cnt += 1
#                        DATA = np.hstack(ldata)
#                        DATA = DATA.transpose()
#
#
#                        true_labels = DATA[:,-1]
#                        DATA = DATA[:,0:-1]
#
#                        # TODO (JZ)
#                        # resultados entregados como string V-Measure y Silhoutte
#                        # incorporar los otros metodos : Ward y affinity
#                        # implementar metodo write_result
#
#                        #snn_experiments(DATA, true_labels)
#                        #kmeans_experiments(DATA, true_labels, 6)
#                        write_results(spectral_experiments(DATA, true_labels, 6))
#
#    '''
#    from sklearn import metrics
#    S = metrics.pairwise.cosine_similarity(newYmat[:,0:-1], newYmat[:,0:-1])
#
#    fig = plt.figure()
#    ax1 = fig.add_subplot(321)
#    ax1.imshow(newYmat[0:10000,0:-1], cmap=plt.cm.Greens, aspect='auto')
#    ax2 = fig.add_subplot(322)
#    ax2.imshow(newYmat[10000:11000,0:-1], cmap=plt.cm.Greys, aspect='auto')
#    ax3 = fig.add_subplot(323)
#    ax3.imshow(newYmat[11000:12000,0:-1], cmap=plt.cm.Reds, aspect='auto')
#    ax4 = fig.add_subplot(324)
#    ax4.imshow(newYmat[12000:13000,0:-1], cmap=plt.cm.Purples, aspect='auto')
#    ax5 = fig.add_subplot(325)
#    ax5.imshow(newYmat[13000:14000,0:-1], cmap=plt.cm.Dark2, aspect='auto')
#    ax6 = fig.add_subplot(326)
#    ax6.imshow(newYmat[14000:15000,0:-1], cmap=plt.cm.Blues, aspect='auto')
#    plt.show()
#    '''
