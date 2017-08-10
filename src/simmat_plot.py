from numpy import genfromtxt
import numpy as np
from matplotlib.pylab import matshow, show, plot, xlim, ylim, cm, savefig
from sklearn import datasets
import sys

def find_cutpoints(Y): # Y is a sorted list of labels
    sYs = []
    v = Y[0]
    cont = 0
    for i in Y:
        if i != v:
            v = i
            sYs.append(cont - 1)
        cont += 1
    return sYs
                    
def get_labels(labelfile):
    f = open(labelfile)
    Y = []
    labels = {}
    for L in f:
        Y.append( labels.setdefault(L.strip(), len(labels) ) )
    f.close()
    return Y, labels

def sort_simmatrix(SS, Y): # the matrix and the corresponding labels (in any order)
    newSS = np.array(SS) # a copy
    newY = list(Y)
    row_offset = 0
    for L in np.unique( Y ):
        for i in np.where( np.array(Y) == L )[0]:
            newSS[row_offset,:] = SS[i,:]
            newY[row_offset] = L
            row_offset += 1
    col_offset = 0
    newnewSS = np.array(newSS) # a copy
    for L in np.unique( Y ):
        for i in np.where( np.array(Y) == L )[0]:
            newnewSS[:,col_offset] = newSS[:, i]
            col_offset += 1
    #dd = np.max( np.triu(newnewSS, 1) )
    #print "Max value found is", dd
    #np.fill_diagonal(newnewSS, dd)
    return newnewSS, newY


def plot_simmat(SS, Y, outfname=None):
    #hold(True)
    matshow(SS, cmap=cm.gist_yarg)
    #matshow(SS, cmap=cm.gray)
    #matshow(SS, cmap=cm.OrRd)
    cpoints = find_cutpoints(Y)
    '''
    for cp in cpoints:
        #Vertical
        plot(np.repeat(cp,SS.shape[0]), range(SS.shape[0]), '-', color="#BAFA85", linewidth=0.01, alpha=0.9)
        #Horizontal
        plot(range(SS.shape[0]), np.repeat(cp,SS.shape[0]), '-', color="#BAFA85", linewidth=0.01, alpha=0.9)
    '''
    xlim(xmin=0)
    xlim(xmax=SS.shape[0])
    ylim(ymin=0)
    ylim(ymax=SS.shape[0])
    if outfname==None:
        show()
    else:
        savefig(outfname)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        if sys.argv[3][-4:] == ".eps":
            X  = genfromtxt(sys.argv[1], delimiter=',')
            Y, _ = get_labels(sys.argv[2])
            nX, nY = sort_simmatrix(X, Y)
            plot_simmat(nX, nY, outfname=sys.argv[3])
        else:
            print "An output file with 'eps' extension must be used."
    else:
        print sys.argv[0]," simmatrix-file row-labels-file output-image-file(.eps)"
