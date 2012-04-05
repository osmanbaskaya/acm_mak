#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Osman Baskaya"

from textprocess import createTextProcessor
from sklearn.preprocessing import Normalizer
from utils import createCF
from numpy import linspace
from scipy.stats import norm
from math import floor
import pickle
import utils
from mediawikifetcher import PATH
from utils import get_matrix_correlation
from mediawikifetcher import DATABASE
from mediawikifetcher import DELIMITER
import pylab as pl


PKL = PATH + 'pkl/'


# Module Functions
def get_ex_instance(filename):
    pkl_file = open(filename, 'rb')
    return pickle.load(pkl_file)

def load_data(filename):
    return get_ex_instance(filename)




def create_category_database():
    mov_dict = dict()

    with open(DATABASE, 'r') as f:
        for line in f.readlines():
            mid, mname, cats = line.split(DELIMITER)
            cats = cats.strip()
            name =  mname.rsplit('(', 1)[0]
            name = name[:len(name)-1]
            name = name.lower()
            mov_dict[name] = cats.split('|')

    cat_database = dict()

    with open(PATH + 'trainset.data') as f:
        
        counter = 0
        for line in f.readlines():
            line = line.strip()
            mname = line.split(',', 1)[1]
            mname = mname[1:]
            try:
                cats = mov_dict[mname]
                for cat in cats:
                    if cat not in cat_database.keys(): 
                        cat_database[cat] = []
                    cat_database[cat].append(counter)
            except:
                pass
            counter += 1
    return cat_database, mov_dict

class Experiment(object):
    
    def __init__(self, setup_mode, corr_func='pearson'):

        self.setup_mode = setup_mode
        self.corr_func = corr_func
        self.cb_prox = None
        self.cf_prox = None

        if setup_mode == 'lightweight':
            self.lightweight_setup()
        elif setup_mode == 'heavyweight':
            self.preprocessing()


    def lightweight_setup(self):
        self.cf_matrix = load_data(PKL + 'cf.pkl')
        self.tf_idf_matrix = load_data(PKL + 'tfidf.pkl')

    def preprocessing(self):
        self.tp = createTextProcessor()
        self.cf_matrix = createCF()
        n = Normalizer(norm='l2', copy=True)
        self.cf_matrix = n.transform(self.cf_matrix) #normalized.

    def create_proximity_matrices(self, offline=True):

        if offline:
            print "Started to create proximity matrices offline"
            self.cf_prox = load_data(PKL + 'cf_prox.pkl')
            self.cb_prox = load_data(PKL + 'cb_prox.pkl')
            #self.cf_prox = load_data(PKL + 'cf_prox_new.pkl')
            #self.cb_prox = load_data(PKL + 'cb_prox_new.pkl')
        else:
            #TODO cf_matrix'in olup olmadigi test edilmedi. yap
            print "Started to create proximity matrices online"
            self.cf_prox = self.cf_matrix * self.cf_matrix.T
            self.cb_prox = self.tf_idf_matrix * self.tf_idf_matrix.T
            #dosyalari da yaz hatta yukardaki isimlerle
        print "Finished to create proximity matrices"

    #def test(self):
        #raise NotImplementedError



class SparsityCorrExperiment(Experiment):
    
    def __init__(self, setup_mode, corr_func='pearson'):
        super(SparsityCorrExperiment, self).__init__(setup_mode, corr_func)

    def test_corr_sparsity(self, draw=False, interval=100):

        #same name like Test
            
        a = self.cb_prox
        b = self.cf_prox

        mat = self.cf_matrix
        m = a.shape[0]

        y = []
        x = []

        low = up = 0
        while 100 >= up:
            up = up + interval
            res = get_matrix_correlation(a, b, matrix=mat, lower=0, upper=up)
            #res = get_matrix_correlation(a, b, matrix=mat, lower=low, upper=up)
            if res is not None:
                if res[1] < 20:
                    continue
                print "Correlation Test between %d - %d" % (low, up) 
                print res
                y.append(res[0][0])
                #x.append((up + low)/2)
                x.append(up)

                #low = up

        if draw:
            utils.drawCorrSparse(x, y)
    

class TopNCorrExperiment(Experiment):
    
    def __init__(self, setup_mode=None, corr_func='pearson'):
        super(TopNCorrExperiment, self).__init__(setup_mode, corr_func)
            
        self.cb_simsorted_dict = None
        self.cf_simsorted_dict = None

    def lightweight_setup(self):
        self.cf_matrix = load_data(PKL + 'cf.pkl')
        self.tf_idf_matrix = load_data(PKL + 'tfidf.pkl')

    def preprocessing(self):
        self.tp = createTextProcessor()
        self.cf_matrix = createCF()
        n = Normalizer(norm='l2', copy=True)
        self.cf_matrix = n.transform(self.cf_matrix) #normalized.

    def create_sorted_dict(self, offline=False):

        if offline:
            #TODO Try-except koymak lazim, dosyalar yok belki
            self.cf_simsorted_dict = load_data(PKL + 'cf_simsorted.pkl')
            self.cb_simsorted_dict = load_data(PKL + 'cb_simsorted.pkl')
        else:
            if self.cb_prox is None or self.cf_prox is None:
                self.create_proximity_matrices()
            self.cb_simsorted_dict = utils.sortSparseMatrix(self.cb_prox)
            print "cb dict has been calculated"
            self.cf_simsorted_dict = utils.sortSparseMatrix(self.cf_prox)
            #utils.write_pickle_obj('cb_simsorted.pkl', self.cb_simsorted_dict)
            #utils.write_pickle_obj('cf_simsorted.pkl', self.cf_simsorted_dict)
            #print "dictionaries are also saved."


    def test_TopN(self, N=20, draw=False):
        
        N = N + 1 # The most similar element is itself.
        if self.cb_simsorted_dict is None or \
                        self.cf_simsorted_dict is None:

            self.create_sorted_dict()

        hits = []
        for movie, neighbors in self.cb_simsorted_dict.iteritems():
            a = set(neighbors[1:N])
            b = set(self.cf_simsorted_dict[movie][1:N])
            common_items = a.intersection(b)
            hits.append(len(common_items))


        filename = "TopN%s.pkl" % N-1
        utils.write_pickle_obj(filename, hits)

        return hits

    @staticmethod
    def get_hit_list(N=20):

        filename = "TopN%d.pkl" % N
        return load_data(filename)
    
    @staticmethod
    def draw_histogram(arr, *args, **kwargs):
        pl.xlabel('Number of Hits')
        pl.ylabel('Number of Items')
        pl.title("Histogram of Hits - Number Of Item")
        pdf, bins, patches = pl.hist(arr, *args, **kwargs)
        pl.axis([0, 15, 0, 2000])
        print 'PDF:', pdf
        print 'Bins:', bins
        print 'Patches:', patches
        pl.grid(True)
        pl.show()



#c, d = create_category_database()
e = TopNCorrExperiment()
hit_list = TopNCorrExperiment.get_hit_list()
TopNCorrExperiment.draw_histogram(hit_list, bins=15)
#e.test_TopN()
#e.create_sorted_dict()

def main():
    ex = Experiment() # Neither heavyweight nor lightweight start
    ex.create_proximity_matrices()
    ex.test_corr_sparsity(draw=True, interval=100)




if __name__ == '__main__':
    main()

