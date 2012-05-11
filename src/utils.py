#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Osman Baskaya"

from mediawikifetcher import PATH
from mediawikifetcher import DELIMITER
from scipy.sparse import lil_matrix, coo_matrix, issparse
from numpy import array, arange
from time import time
from scipy.stats import pearsonr, kendalltau, spearmanr, chisquare
from numpy import tril_indices
from sys import stderr
from random import randint
import pylab as pl
import numpy as np

def writedict2file(d, filename='dict.dat'):

    text = []
    with open(filename, 'w') as f:
        for key, value in d.iteritems():
            text.append(str(key) + ' : ' + str(value) + '\n')

        f.write(''.join(text))

def writearray2file(a, filename='array.dat'):
    text = []

    with open(filename, 'w') as f:
        for i, num in enumerate(a):
            line = "%d, %f\n"
            text.append(line % (i, num))

        f.write(''.join(text))


def createCF(path=PATH, filename='ratings2.dat'):
    start_time = time()
    m, n = get_dataset_size()
    #user_item_matrix = coo_matrix((m,n), dtype=uint8)
    row = []
    column = []
    data = []

    with open(path + filename) as f:
        for line in f.readlines():
            user_id, mov_id, rat, t = line.split('::')
            column.append(int(user_id)-1)
            row.append(int(mov_id))
            data.append(float(rat))
            #user_item_matrix[int(user_id)-1, int(mov_id)] = rat

    row = array(row)
    column = array(column)
    data = array(data)
    print 'CF matrix has been created. ' + \
            'It tooks %s seconds' % (time() - start_time)
    return coo_matrix((data, (row, column)), shape=(n,m))



def get_dataset_size(path=PATH, filename='ratings2.dat'):
    user_set = set()
    item_set = set()
    with open(path + filename) as f:
        for line in f.readlines():
            user_id, mov_id, rat, t = line.split('::')
            user_set.add(user_id)
            item_set.add(mov_id)

    return (len(user_set), len(item_set))


def get_matrix_correlation(cb_prox, cf_prox, lower=None, upper=None, 
                    matrix=None, narrow=True, corr_method='pearson'):

    if matrix is None:
        cb = cb_prox
        cf = cf_prox
    else:
        index_list = get_itemID_between_intervals(matrix, lower, upper)
        if len(index_list) <= 1:
                return None 
        cb = cb_prox[index_list, :]
        cf = cf_prox[index_list, :]
        if narrow:
            cb = cb[:, index_list]
            cf = cf[:, index_list]

    r, c = cb.shape


    ind = tril_indices(r, k=-1)
    
    #m = cb.toarray().reshape(r*c, 1)
    #n = cf.toarray().reshape(r*c, 1)

                
    m = cb.toarray()[ind]
    n = cf.toarray()[ind]


    if corr_method == 'pearson':
        func = pearsonr
    elif corr_method == 'kendall':
        func = kendalltau
    elif corr_method == 'spearman':
        func = spearmanr
    elif corr_method == 'chisquare':
        func = chisquare
    else: print "there is no such corr function named %s\n" % corr_method


    try:
        ret = (func(m,n), r) # Result, number of item (r)
    except RuntimeWarning:
        pass
    return ret

def get_itemID_between_intervals(matrix, lower=None, upper=None):

    if not issparse(matrix):
        stderr.write("matrix should be sparse.\n")
        exit(1)
        
    mat = matrix.tolil()
    m, n = mat.shape

    if lower is None:
        lower = 1

    if upper is None:
        upper = n

    if lower > upper:
        lower, upper = upper, lower

    lower =  [item_id for item_id in range(m) 
                            if mat.getrow(item_id).size >= lower]
    upper =  [item_id for item_id in range(m) 
                            if mat.getrow(item_id).size >= upper+1]
    lower = set(lower)
    upper = set(upper)
    return list(lower.difference(upper))


def drawCorrSparse(x, y):
    pl.title("Pearson Results - Number of Ratings that Movies have")
    pl.xlabel("Number of rating that movies have")
    pl.ylabel("Pearson Result")
    pl.grid(True)

    #m = len(x)
    #pl.yticks(arange(m) + y[0])
    #pl.xticks(arange(m) + x[0])
    pl.scatter(x, y)
    pl.show()


def create_random_ratings_file(filename, delimiter=DELIMITER):
    
    g = open(PATH + 'ratings3.dat', 'w')

    text_list = []

    with open(PATH + filename) as f:
        for line in f.readlines():
            p_rat = randint(1, 5)
            a = line.split(delimiter)
            a[2] = str(p_rat)
            text_list.extend(a)

    g.write('::'.join(text_list))
    g.close()


def get_synthetic_spmatrix(matrix):
    

    #TODO control whether matrix mat sparse or not.
    mat = matrix.tocoo()
    mat_ind = matrix.tolil()
    m = mat.size
    col = mat.col
    row = mat.row
    s = mat.shape[0]
    num_swap =  s*s/10
    print num_swap

    for i in xrange(num_swap):
        
        e1 = randint(0, m-1)
        #e2 = randint(0, m-1)
        overlap = True
        count = 1 
        while overlap:
            count = count + 1
            e2 = randint(0, m-1)
            pair1 = (row[e1], col[e2],)
            pair2 = (row[e2], col[e1],)
            if count > mat.shape[0]: # bu satirda duzgun bir degisim olmayacak
                e1 = randint(0, m-1)
                i = i + 1
                print 'prob'
                count = 1
            if mat_ind[pair1] == 0 and mat_ind[pair2] == 0:
                overlap = False
                mat_ind[pair1], mat_ind[pair2] = mat_ind[pair2], mat_ind[pair1]
        if i % 100000 == 0:
            print i

        pair1 = (row[e1], col[e1],)
        pair2 = (row[e2], col[e2],)
        #print mat_ind[pair1], mat_ind[pair2]
        mat_ind[pair1] = 0; mat_ind[pair2] = 0
        #print mat_ind[pair1], mat_ind[pair2]
        col[e1], col[e2] = col[e2], col[e1]
        #print col
        #print row

    return coo_matrix((mat.data, (row, col)), shape=mat.shape)


def test_get_synthetic():
    #import numpy as np
    #a = np.matrix(np.random.rand(5,5))
    from scipy.sparse import csr_matrix
    from experiment import Experiment, PKL
    a = Experiment.load_data(PKL + 'cf.pkl')
    print "Load finish"
    #m, n = a.shape
    #siz = m*n
    #for i in xrange(siz + siz/2):
        #e1= randint(0, m-1)
        #e2= randint(0, m-1)
        #a[e1, e2] = 0.


    b = csr_matrix(a)
    #print b.todense()
    print "Test start"
    k = get_synthetic_spmatrix(b)
    return k, a


def load_sparse_data(filename):
    #from numpy import matrix
    data = []
    row = []
    col = []
    with open(PATH + filename) as f:
        
        line_shape = f.readline()
        m, n = line_shape.split()
        f.readline() # gereksiz bir satir oldugundan oku, bir sey yapma

        for line in f.readlines():
            r, c, v = line.split()
            r = int(r)
            c = int(c)
            v = float(v)
            row.append(r)
            col.append(c)
            data.append(v)

            #data.append([r,c,v])

    #mat = matrix(data)
    #col = mat[:,1]
    #row = mat[:,0]
    #data = mat[:,2]
    m = int(m)
    n = int(n)

    #return mat

    return coo_matrix((data, (row, col)), shape=(m, n))

def sortSparseMatrix(m, rev=True, only_indices=True):

    """ Sort a sparse matrix and return column index dictionary
    """
    col_dict = dict() 
    for i in xrange(m.shape[0]): # assume m is square matrix.
        d = m.getrow(i)
        s = zip(d.indices, d.data)
        sorted_s = sorted(s, key=lambda v: v[1], reverse=True)
        if only_indices:
            col_dict[i] = [element[0] for element in sorted_s]
        else:
            col_dict[i] = sorted_s
    return col_dict


def write_pickle_obj(filename, obj):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def getTopN(mat, r, N=20, rev=True):
    """
        mat = sparse matrix
        r = row number
        N = Number of id
    """

def main():
    #createCF()
    a, k = test_get_synthetic()
    a.save('syntetic_cf.dat')


if __name__ == '__main__':
    main()

