#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Osman Baskaya"

from textprocess import createTextProcessor
from sklearn.preprocessing import Normalizer
from utils import createCF
from numpy import linspace
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import jaccard
from math import floor
import utils
from mediawikifetcher import PATH
from utils import create_category_matrix
from utils import get_matrix_correlation, load_data
from mediawikifetcher import DATABASE
from mediawikifetcher import DELIMITER
import pylab as pl
import cf_models


PKL = PATH + 'pkl/'



def create_category_database():

    # Very Fast. Do not need to use pickle or etc.

    mov_dict = dict()

    with open(DATABASE, 'r') as f:
        for line in f.readlines():
            mid, mname, cats = line.split(DELIMITER)
            cats = cats.strip()
            name =  mname.split('(', 1)[0]
            name = name[:len(name)-1]
            name = name.lower()
            mov_dict[name] = set(cats.split('|'))

    cat_database = dict()

    with open(PATH + 'trainset.data') as f:
        
        counter = 0
        movid_dict = dict()
        for line in f.readlines():
            line = line.strip()
            mname = line.split(',', 1)[1]
            mname = mname[1:]
            try:
                cats = mov_dict[mname]
                movid_dict[counter] = cats
                for cat in cats:
                    if cat not in cat_database.keys(): 
                        cat_database[cat] = set()
                    cat_database[cat].add(counter)
            except:
                pass
                
            counter += 1
    return cat_database, movid_dict

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


    def create_proximity_matrices(self, ptype='all', offline=True):

        #ptype = proximity matrix type
        #TODO Cok fena uzun yazilmis... Sonra duzelt.

        ptype = ptype.lower()
        if ptype == 'all':
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
        elif ptype == 'cb':
            if offline:
                print "Started to create proximity matrices offline for CB"
                self.cb_prox = load_data(PKL + 'cb_prox.pkl')
                #self.cb_prox = load_data(PKL + 'cb_prox_new.pkl')
            else:
                print "Started to create proximity matrices online"
                self.cb_prox = self.tf_idf_matrix * self.tf_idf_matrix.T
            print "Finished to create proximity matrices"
        elif ptype == 'cf':
            if offline:
                print "Started to create proximity matrices offline for CF"
                self.cf_prox = load_data(PKL + 'cf_prox.pkl')
                #self.cf_prox = load_data(PKL + 'cf_prox_new.pkl')
            else:
                self.cf_prox = self.cf_matrix * self.cf_matrix.T
                print "Started to create proximity matrices online"
            print "Finished to create proximity matrices"


class SparsityCorrExperiment(Experiment):
    
    def __init__(self, setup_mode, corr_func='pearson'):
        super(SparsityCorrExperiment, self).__init__(setup_mode, corr_func)

    def test_corr_sparsity(self, draw=False, interval=100, min_item=20):

        #same name like Test
            
        a = self.cb_prox
        b = self.cf_prox

        mat = self.cf_matrix
        m = a.shape[0]

        y = []
        x = []

        low = up = 0
        while m >= up:
            up = up + interval
            res = get_matrix_correlation(a, b, matrix=mat, lower=0, upper=up)
            #res = get_matrix_correlation(a, b, matrix=mat, lower=low, upper=up)
            if res is not None:
                if res[1] < min_item:
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


    def create_sorted_dict(self, offline=False):

        if offline:
            # Offline cok zaman aliyor, gerek yok gibi bir sey.
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


    def take_hits(self, N, movies=None):

        N = N + 1 # The most similar element is itself.

        if movies is None:
            movies = self.cb_simsorted_dict.keys()

        hits = []
        for movie in movies:
            n1 = set(self.cb_simsorted_dict[movie][1:N])
            n2 = set(self.cf_simsorted_dict[movie][1:N])

            common_items = n1.intersection(n2)
            hits.append(len(common_items))
        
        return hits




    def test_TopN(self, N=100):
        
        if self.cb_simsorted_dict is None or \
                        self.cf_simsorted_dict is None:

            self.create_sorted_dict()

        hits = self.take_hits(N=N) 
        filename = PKL + "TopN%d.pkl" % (N)
        utils.write_pickle_obj(filename, hits)

        return hits


    def test_TopN_intervals(self, N=100, interval=100, min_item=20):
        

        mat = self.cf_matrix

        m = mat.shape[0] # 3417x6000
        low = up = 0
        
        values = []
        inters = []
        while m >= up:
            up = up + interval
            movies_compare = utils.get_itemID_between_intervals(mat, low, up)
            num_movies = len(movies_compare)
            if num_movies >= min_item:
                print "%d - %d | Number of Movies: %d" % (low, up, num_movies)
                mean = sum(self.take_hits(N=N, movies=movies_compare)) / float(num_movies)
                values.append(mean)
                print mean
                inters.append((low+up)/2)

                low = up
        TopNCorrExperiment.draw_topN_corr(inters, values, N=N)


    @staticmethod
    def get_hit_list(N=100):

        filename = PKL + "TopN%d.pkl" % N
        try:
            return load_data(filename)
        except IOError:
            e = TopNCorrExperiment()
            hit_list = e.test_TopN(N=N)
            return hit_list


    
    
    @staticmethod
    def draw_topN_corr(arr1, arr2, N=100, *args, **kwargs):
        pl.ylabel('Mean value of #Hits (Common Elements)')
        pl.xlabel('Average value of High and Low interval values')
        pl.title("Top%d of Hits - Number Of Item" % N)
        pl.grid(True)
        pl.scatter(arr1, arr2, *args, **kwargs)
        pl.show()
        

    @staticmethod
    def draw_histogram(arr, N, *args, **kwargs):
        pl.xlabel('Number of Hits')
        pl.ylabel('Number of Items')
        pl.title("Histogram of Hits - %d Number Of Item" % N)
        pdf, bins, patches = pl.hist(arr, bins=xrange(N+2), *args, **kwargs)
        #pl.axis([0, 15, 0, 2000])
        #E = (np.array(pdf) * np.array(bins)) / float(N)
        E = (pdf * bins[:-1]).sum() / float(pdf.sum())
        print 'PDF:', pdf
        print 'Bins:', bins
        print 'Expected Value %f' % E
        pl.grid(True)
        pl.show()



class CategoryCorrExperiment(Experiment):


    """ This experiment is for [Genre - Rating Correlation] and 
    [Genre - Content Correlation]
     
     """

    def __init__(self, setup_mode=None, approach='cf', corr_func='pearson'):
        self.sim_dict = None
        self.approach = approach.lower() # This indicate which approach is tested | cf or cb
        super(CategoryCorrExperiment, self).__init__(setup_mode, corr_func)

    def lightweight_setup(self):
        if self.approach == 'cf':
            pass
            #self.cf_matrix = load_data(PKL + 'cf.pkl')
        elif self.approach == 'cb':
            pass
            
        
        self.create_approach_matrix()
        self.cat_mat = create_category_matrix()


    def create_approach_matrix(self):

        if self.sim_dict is None:
            if self.approach == 'cf': 
                self.create_proximity_matrices(ptype='cf')
                self.sim_dict = utils.sortSparseMatrix(self.cf_prox)
                print "Similarity dict has been created for %s" % self.approach
            elif self.approach == 'cb':
                self.create_proximity_matrices(ptype='cb')
                self.sim_dict = utils.sortSparseMatrix(self.cb_prox)
                print "Similarity dict has been created for %s" % self.approach
            else:
                print "There is no approach: %s" % self.approach
                print "Please pick <cf> or <cb>"
                exit(-1)


    def take_hits(self, cat_data, mid_dict, N, movies=None):

        N = N + 1 # The most similar element is itself.
        
        #total_hits = []

        if movies is None:
            movies = mid_dict.keys()
        
        m = len(movies)
        total = 0
        jac_list = []
        for movid in movies:
            
            jac = 0 # jaccard score of movies
            
            mov_genre = self.cat_mat[movid]
            neighbors = set(self.sim_dict[movid][1:N])
            for neighbor in neighbors:
                n_genre = self.cat_mat[neighbor]
                jac = jac + jaccard(mov_genre, n_genre)

            jac = jac / N # Avg Jaccard of the movie.
            jac_list.append(jac) # for histogram
            
            total = total + jac
            
            
            #total_hits.append(n)

        total = total / m # Avg jaccard of movies.
        return (total, jac_list)

    def test_category_accuracy(self, N=100):

        cat_data, mid_dict = create_category_database()
        avgJacMov, jac_list = self.take_hits(cat_data, mid_dict, N)
        results = [avgJacMov, jac_list]
        #filename = PKL + "Category%dfor%s.pkl" % (N, self.approach.upper())
        #utils.write_pickle_obj(filename, results)

        return results

    def test_category_accuracy_interval(self, interval=100, N=100, min_item=20):

        # If harsh is "True" then every genre(s) of a movie should be fitted
        # to other movie.

        cat_data, mid_dict = create_category_database()
        
        try:
            mat = self.cf_matrix
        except AttributeError:
            self.lightweight_setup()
            mat = self.cf_matrix

        m = mat.shape[0]

        low = up = 0
        
        values = []
        inters = []
        while m >= up:
            up = up + interval
            movies_compare = utils.get_itemID_between_intervals(mat, low, up)
            num_movies = len(movies_compare)
            if num_movies >= min_item:
                print "%d - %d | Number of Movies: %d" % (low, up, num_movies)

                [total, jac_list] = self.take_hits(cat_data, mid_dict, N=N, movies=movies_compare)
                mean = total

                values.append(mean)
                print mean
                inters.append((low+up)/2)
                low = up

        CategoryCorrExperiment.draw_catN_corr(inters, values, N=N)
        return (inters, values)
    
    @staticmethod
    def get_hit_list(N=100):

        filename = PKL + "Category%d.pkl" % N
        return load_data(filename)

    @staticmethod
    def draw_histogram(arr, N, approach, *args, **kwargs):
        pl.xlabel('Jaccard Distance')
        pl.ylabel('Number of Items')
        pl.title("Histogram of Jaccard Distance and Number of Movies [N = %d, %s]" % (N, approach))
        pdf, bins, patches = pl.hist(arr, bins=np.linspace(0, 1, 11), *args, **kwargs)
        #pl.axis([0, 15, 0, 2000])
        #E = (np.array(pdf) * np.array(bins)) / float(N)
        E = (pdf * bins[:-1]).sum() / float(pdf.sum())
        print 'PDF:', pdf
        print 'Bins:', bins
        print 'Expected Value %f' % E
        pl.grid(True)
        pl.show()

    @staticmethod
    def draw_catN_corr(arr1, arr2, N=100, *args, **kwargs):
        pl.ylabel('Mean value of Jaccard Distance')
        pl.xlabel('Average value of High and Low interval values')
        #pl.title("Category Correlation [%d of Hits]" % N)
        pl.title("Category Analysis [N = %d]" % N)
        #pl.axis(0, max(arr2)+1, 0, 2500)
        pl.grid(True)
        pl.scatter(arr1, arr2, *args, **kwargs)
        pl.show()

class FanHypoExperiment(object):


    def __init__(self, setup_mode='lightweight'):



        self.cf_matrix = None
        self.movie_dict = dict()
        self.user_dict = dict()
        self.test_set = dict(FanHypoExperiment.get_movie_genre_dict())

        if setup_mode == 'lightweight':
            self.lightweight_setup()

    def lightweight_setup(self):
        self.cf_matrix = load_data(PKL + 'cf.pkl')

    def test_fanHypo(self, lower, upper):

        """
            Test of fan hypothesis. This method assumes that an unpopular 
            movie has at most number of `up` raters.

        """

        cf_models.preprocessing()
        if self.cf_matrix is None:
            print "Lightweight setup should be needed. It is calling now"
            self.lightweight_setup()

        cf_mat = self.cf_matrix.tolil() # for efficient indexing
        self.prepare_test_set(lower, upper)

        print "Number of test examples in %d-%d is %d\n" % \
                                            (lower,upper,len(self.test_set))

        numerator = denominator = 0
        for movie_id in self.test_set:
            mov = cf_models.Movie.get_movie_by_ID(movie_id, cf_mat)
            #print "Movie ID: %s, Fan Dist: %s" % (mov, mov.fan_dist)
            #print mov.matchscore, "Actual Genres are", mov.actual_genres
            numerator += mov.matchscore[0]
            denominator += mov.matchscore[1]

        print "\nNumber of Fan %d, Number of User %d\n" % (numerator, denominator)
        print "Total Score is %f\n" % (float(numerator)/denominator)
            

        cf_models.update_the_dbs()


    def prepare_test_set(self, lower, upper):
        m = utils.get_itemID_between_intervals(self.cf_matrix, \
                                                lower=lower, upper=upper)
        s = self.test_set.keys()
        removes = set(s).difference(m)
        for element in removes:
            self.test_set.pop(element)



    @staticmethod
    def get_movie_genre_dict():
        cat_database, movie_genre_dict = create_category_database()
        return movie_genre_dict




#c, d = create_category_database()
#e = TopNCorrExperiment(setup_mode='lightweight')
#e = TopNCorrExperiment()
#hit_list = e.test_TopN()
#hit_list = TopNCorrExperiment.get_hit_list()
#TopNCorrExperiment.draw_histogram(hit_list, bins=15)
#e.create_sorted_dict()


# Main-like Functions
########################################
def fanhypo():
    f = FanHypoExperiment()
    f.test_fanHypo(lower=1, upper=5)

def category_interval():
    n = 20
    e = CategoryCorrExperiment(setup_mode='lightweight', approach='cf')
    e.test_category_accuracy_interval(N=n)


def category_all():
    n = 100
    e = CategoryCorrExperiment(setup_mode='lightweight', approach='cb')
    [avgJacTotal, jac_list] = e.test_category_accuracy(N=n)
    #hit_list = CategoryCorrExperiment.get_hit_list(N=n, harsh=h)
    print "Average Jaccard Distance of Genre-Wiki = %f [N=%d]" % (avgJacTotal, n)
    CategoryCorrExperiment.draw_histogram(jac_list, N=n, approach=e.approach)

def topN_interval():
    n = 20
    e = TopNCorrExperiment(setup_mode='lightweight')
    e.create_sorted_dict()
    e.test_TopN_intervals(N=n)

def topN_all():

    n = 20
    #e = TopNCorrExperiment()
    #hit_list = e.test_TopN(N=n)
    hit_list = TopNCorrExperiment.get_hit_list(N=n)
    TopNCorrExperiment.draw_histogram(hit_list, N=n)

def main():
    #ex = Experiment() # Neither heavyweight nor lightweight start
    #ex.create_proximity_matrices()
    #ex.test_corr_sparsity(draw=True, interval=100)

    #topN_all()
    #topN_interval()
    #category_interval()
    category_all()
    #fanhypo()


if __name__ == '__main__':
    main()

