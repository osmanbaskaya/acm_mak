#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Osman Baskaya"


import unittest
from experiment import PKL
import experiment
#from textprocess import createTextProcessor
from utils import createCF, load_sparse_data
from sklearn.preprocessing import Normalizer

class ExperimentTests(unittest.TestCase):
    

    def setUp(self):
        pass



    #def test_ver2_rand(self):
        #self.ex = experiment.Experiment()
        #self.ex.cf_matrix = createCF(filename='ratings3.dat')

        #n = Normalizer(norm='l2', copy=True)
        #self.ex.cf_matrix = n.transform(self.ex.cf_matrix) #normalized.
        #self.ex.cb_prox = experiment.Experiment.load_data(PKL + 'cb_prox.pkl')
        #self.ex.cf_prox = self.ex.cf_matrix * self.ex.cf_matrix.T
        #self.ex.test_corr_sparsity(draw=True, interval=100)

    def test_ver2_syntetic_dataset(self):

        self.ex = experiment.Experiment()
        self.ex.cf_matrix = load_sparse_data('syntetic_cf.dat')
        n = Normalizer(norm='l2', copy=True)
        self.ex.cf_matrix = n.transform(self.ex.cf_matrix) #normalized.
        self.ex.cb_prox = experiment.Experiment.load_data(PKL + 'cb_prox.pkl')
        self.ex.cf_prox = self.ex.cf_matrix * self.ex.cf_matrix.T
        self.ex.test_corr_sparsity(draw=True, interval=100)


def main():
    unittest.main()

if __name__ == '__main__':
    main()

