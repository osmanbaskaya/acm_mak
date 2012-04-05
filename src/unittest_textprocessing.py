#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Osman Baskaya"


import unittest
from textprocess import TextProcessor
from wiki import get_all_offline_articles
import string
import itertools
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from operator import itemgetter



class TextProcessingTests(unittest.TestCase):
    

    def setUp(self):
        all_articles = get_all_offline_articles()
        self.tp =  TextProcessor(all_articles)




    def test_preprocessing(self):
        """ Preprocessing consists of couple of steps such as 
            punctuation removing. This method tests that all
            punctuations are removed.
        """

        punctuations = string.punctuation
        #print "all punctuations are %s" % punctuations
        self.tp.preprocessing()
        product = itertools.product(punctuations, self.tp.trainset)
        no_punctuation = True
        for p, article in product:
            #print p, article
            if p in article.text:
                no_punctuation = False
                break
        self.failUnless(no_punctuation == True)



    def test_transformation(self):

        #TODO: Remove this function. Useless now

        train_set = ("The sky is blue.", "The sun is bright.")
        #test_set = ("The sun in the sky is bright.",
                    #"We can see the shining sun, the bright sun.")
        count_vectorizer = CountVectorizer('english')
        count_vectorizer.fit_transform(train_set)


    def test_cosine(self):
        #mov1 = 2910 # starwars a new hope
        #mov1 = 1948 # matrix
        mov1 = 1319 # matrix
        #mov1 = 3317 # when harry met sally
        self.tp.preprocessing()
        self.tp.create_vocabulary()
        self.tp.transformation()
        self.tp.calculate_idfs()
        self.tp.build_tfidf_matrix()
        # Movie which compare every movies with
        m1 = self.tp.tf_idf_matrix.getrow(mov1)
        sims = []
        for i in range(len(self.tp.trainset)):
            m2 = self.tp.tf_idf_matrix.getrow(i)
            cos_sim = (m1 * np.transpose(m2)).todense()
            cos_sim = cos_sim[0,0]
            if cos_sim > 0.07:
                sims.append([cos_sim, self.tp.trainset[i].title])

        sims = sorted(sims, key=itemgetter(0), reverse=True)
        print_m(sims[:15])
        leng = len(sims)
        print "\n\n"
        print_m(sims[leng-10:leng])

    def test_similar_items(self):
        pass

def print_m(m):
    for element in m:
        print element

def main():
    unittest.main()

if __name__ == '__main__':
    main()

