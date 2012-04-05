#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Osman Baskaya"

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from Stemmer import Stemmer
from wiki import get_all_offline_articles
import string
from copy import deepcopy
from time import time
from operator import attrgetter
from mediawikifetcher import DATABASE
from mediawikifetcher import PATH
from mediawikifetcher import DELIMITER
from scipy.sparse import coo_matrix
import numpy as np
#from mediawikifetcher import read_movie_database
import os

class Processor(object):
    
    def __init__(self, data):
        self.data = data
        


class TextProcessor(Processor):
    

    def __init__(self, data=get_all_offline_articles(), language='english', stop_words='english'):
        super(TextProcessor, self).__init__(data)
        #self.trainset = deepcopy(data.values()[0:1])
        self.trainset = deepcopy(data.values())
        # Sorting by title
        self.trainset = sorted(self.trainset, key=attrgetter('title'))

        self.data = data
        #analyzer = WordNGramAnalyzer(stop_words=ENGLISH_STOP_WORDS)
        #self.count_vectorizer = CountVectorizer(analyzer=analyzer)
        self.count_vectorizer = CountVectorizer(stop_words=stop_words)

        self.tfidf = TfidfTransformer(norm="l2")
        self.stemmer = Stemmer(language)


    def stemming(self, article):

        stem_list = map(self.stemmer.stemWord, article.text.split())
        article.text =  ' '.join(stem_list)


    # Thanks to Chris Perkins & Raymond Hettinger
    @staticmethod
    def translator(frm='', to='', delete='', keep=None):
        if len(to) == 1:
            to = to * len(frm)
        trans = string.maketrans(frm, to)
        if keep is not None:
            allchars = string.maketrans('', '')
            delete = allchars.translate(allchars, keep.translate(allchars, delete))
        def translate(s):
            return s.translate(trans, delete)
        return translate


    def preprocessing(self):
        start_time = time()
        punc_remove = TextProcessor.translator(delete=string.punctuation)
        #self.trainset = map(punc_remove, self.trainset)
        for article in self.trainset:
            article.text = punc_remove(article.text)
            self.stemming(article)

        print 'Preprocessing has been finished. ' + \
                'It tooks %s seconds' % (time() - start_time)

    def create_vocabulary(self):
        start_time = time()
        self.count_vectorizer.fit_transform([article.text \
                                    for article in self.trainset])

        print 'Vocabulary creation has been finished. It tooks %s seconds' \
                                    % (time() - start_time)


    def transformation(self):
        start_time = time()
        #self.freq_term_matrix = self.count_vectorizer.transform([article.text \
                                            #for article in self.trainset])

        c = self.count_vectorizer.transform([article.text \
                                            for article in self.trainset])

        new_data = np.log(c.data+3)/np.log(4) 
        self.freq_term_matrix = coo_matrix((new_data, 
                                            (c.row, c.col)), shape=c.shape)
        print 'Transformation has been finished. ' + \
                'It tooks %s seconds' % (time() - start_time)



    def calculate_idfs(self, norm="l2"):
        
        start_time = time()
        self.tfidf = TfidfTransformer(norm)
        self.tfidf.fit(self.freq_term_matrix)

        print 'IDF Calculation has been finished. ' + \
                'It tooks %s seconds' % (time() - start_time)

    def build_tfidf_matrix(self):
        start_time = time()
        self.tf_idf_matrix = self.tfidf.transform(self.freq_term_matrix)
        print 'TF/IDF Matrix has been created. ' + \
                'It tooks %s seconds' % (time() - start_time)


    def writetrainset2file(self):
        with open('trainset.data', 'w') as f:
            for i, movie in enumerate(self.trainset):
                line = "%d, %s\n"
                f.write(line % (i, movie))
    
    def fit_ids(self, ml_movies=DATABASE, ml_ratings=PATH+'1m/ratings.dat'):
        
        r = create_ratings_dict(ml_ratings)
        m = dict() 
        
        with open(ml_movies) as f:
            for element in f.readlines():
                element = element.split(DELIMITER)
                m_id, movie_name = element[0], element[1]
                movie_name = movie_name.split(' (', 1)[0]
                m[movie_name.lower()] = m_id 


        f = open('ratings2.dat', 'w')
        counter = 0
        movie_list = []
        remove_list = []
        for i, mov in enumerate(self.trainset):
            m_id = m[mov.title]

            #print i, mov, m_id
            try:
                for rating in r[m_id]:
                    rating[1] = str(i)
                    movie_list.extend(rating)
            except KeyError:
                print i, mov.title, m_id, "filmine hic oy atilmamis"
                counter += 1
                remove_list.append(mov.title.title() +  '.txt')


        f.write('::'.join(movie_list))
        f.close()
        remove_files(remove_list)
        print "Number of unrated movies:", counter

    def prepare(self):
        self.preprocessing()
        self.create_vocabulary()
        self.invVoc = sorted(self.count_vectorizer.vocabulary_.keys())
        self.transformation()
        self.calculate_idfs()
        self.build_tfidf_matrix()


    def get_words_from_vector(self, article_id):
        words = dict()
        c = self.freq_term_matrix.getrow(article_id)
        index = c.nonzero()[1]
        for  i in index:
            word = self.invVoc[i]
            words[word] = c[0,i]
        return words

    def compare_two_article(self, art1, art2):
        d = self.get_words_from_vector(art1)
        e = self.get_words_from_vector(art2)
        dd = set(d.keys())
        ed = set(e.keys())
        return dd.intersection(ed)


def remove_files(remove_list, path="movies/"):
    all_files = os.listdir(path) 
    os.chdir(path)
    for file_name in remove_list:
        try:
            os.remove(file_name)
            print '%s has been removed' % file_name
        except OSError:
            for element in all_files:
                if file_name.lower() == element.lower():
                    file_name = element
                    break
            os.remove(file_name)
            #print '%s has not been removed' % file_name
    

def create_ratings_dict(ml_ratings, delimiter=DELIMITER):
    r = dict()
    with open(ml_ratings, 'r') as f:
        for line in f.readlines():
            elements = line.split(delimiter)
            try:
                r[elements[1]].append(elements)
            except KeyError:
                r[elements[1]] = [elements]

    return r



def createTextProcessor():

    all_articles = get_all_offline_articles()
    tp = TextProcessor(all_articles)
    tp.prepare()
    return tp

#print self.freq



def main():
    pass
    #all_articles = get_all_offline_articles()
    #tp = TextProcessor(all_articles)
    #tp.fit_ids()
    #tp.writetrainset2file()

if __name__ == '__main__':
    main()

