#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Osman Baskaya"



from operator import itemgetter
from experiment import PKL, load_data
import utils
from experiment import FanHypoExperiment



""" This module provides some concepts such as User, Movie
    in collaborative filtering approach.
"""

db_name = "1M"
movie_pkl_file_name = PKL + db_name + "_fan_hypo_movie.pkl"
user_pkl_file_name = PKL + db_name + "_fan_hypo_user.pkl"
movie_dict = {}
user_dict = {}
_moviedict_update = False
_userdict_update = False
movie_genre_dict = {} 


def _create_movgenre_dict():

    global movie_genre_dict
    movie_genre_dict = FanHypoExperiment.get_movie_genre_dict()
    

def _load_dbs():
    global user_dict
    global movie_dict
    try:
        movie_dict = load_data(movie_pkl_file_name)
    except IOError:
        print 'there is no pkl file named %s' % movie_pkl_file_name
    try: 
        user_dict = load_data(user_pkl_file_name)
    except IOError:
        print 'there is no pkl file named %s' % user_pkl_file_name

def preprocessing():
    _load_dbs()
    _create_movgenre_dict()


class User(object):
    
    def __init__(self, uid, cf_mat):

        """
        It should not be connected directly because of redundant
        processing. Instead, movie should be created by get_movie_by_ID
        method.

        """
        self.uid = uid
        self.__fav_genre = None # the most watched movie genre
        self.__movie_genres = dict() # number of movies which this user 
                                     # watched for each genre
        u = cf_mat.getcol(uid)
        u_m = u.nonzero()
        self.movies = u_m[0] # first col contains the ids'

    @staticmethod
    def get_user_by_ID(uid, cf_mat):
        if uid not in user_dict:
            return User(uid, cf_mat)
        else:
            return user_dict[uid]

    @property
    def fav_genre(self):
        if self.__fav_genre is None:
            self.__fav_genre = self.calculate_fav_genre()

        if self.uid not in user_dict:
            user_dict[self.uid] = self
            global _userdict_update
            _userdict_update = True

        return self.__fav_genre

    @property
    def movie_genres(self):
        if not self.__movie_genres:
            self.__prepare_movie_genres()

        return self.__movie_genres

    def calculate_fav_genre(self):
        if not self.__movie_genres:
            self.__prepare_movie_genres()

        return self.__movie_genres[0][0]

    def __prepare_movie_genres(self):
        mg = self.get_genres()
        self.__movie_genres = self.__get_genre_dist(mg)
        self.__movie_genres = sorted(self.__movie_genres.iteritems(), \
                            key=itemgetter(1), reverse=True)
        return self.__movie_genres

    def get_genres(self):
        m = dict(movie_genre_dict)
        s = set(self.movies)
        removes = set(m.keys()).difference(s)
        for element in removes:
            m.pop(element)

        return m

    def __get_genre_dist(self, mg):
        d = dict() 
        for genres in mg.itervalues():
            for genre in genres:
                if genre not in d:
                    d[genre] = 1
                else:
                    d[genre] += 1

        return d

    def __repr__(self):
        
        return "<User Object: uid=%d>" % self.uid


class Movie(object):
    
    def __init__(self, mid, cf_mat):

        """
        It should not be connected directly because of redundant
        processing. Instead, movie should be created by get_movie_by_ID
        method.

        """

        self.actual_genres = movie_genre_dict[mid]
        self.mid = mid
        self.fan_dist = []
        self.cf_mat = cf_mat

        m = cf_mat.getrow(mid) 
        m_u = m.nonzero() # list of users who rate that movie.
        self.users = m_u[1] # second col contains the ids'
        self.fan_dist = None
        self.matchscore = None
        self.__do_all_processing()

    

    def __do_all_processing(self):
        

        self.most_rated_genre_by_users
        self.matchscore = self.__get_num_of_match() #matchscore

        # Write to the database
        movie_dict[self.mid] = self
        global _moviedict_update
        _moviedict_update = True



    @staticmethod
    def get_movie_by_ID(mid, cf_mat):
        if mid not in movie_dict:
            return Movie(mid, cf_mat)
        else:
            return movie_dict[mid]


    @property
    def most_rated_genre_by_users(self):
        
        if not self.fan_dist:
            self.fan_dist = self.__get_fan_dist()


        f = sorted(self.fan_dist.iteritems(), key=itemgetter(1), reverse=True)

        return f[0][0]


    def __get_num_of_match(self):
        
        """ Number of users who are the fan of mov's actual genre """

        hits = 0
        for genre in self.actual_genres:
            if genre in self.fan_dist:
                hits += self.fan_dist[genre]

        return (hits, len(self.users))

    def __get_fan_dist(self):
        
        fan_dist = dict()

        for user_id in self.users:
            user = User.get_user_by_ID(user_id, self.cf_mat)
            fav_genre = user.fav_genre
            if fav_genre not in fan_dist.keys():
                fan_dist[fav_genre] = 1
            else:
                fan_dist[fav_genre] += 1

        return fan_dist

        #return sorted(fan_dist.iteritems(), \
                            #key=itemgetter(1), reverse=True)


    def __repr__(self):
        
        return "<Movie object: mid=%d>" % self.mid


def update_the_dbs():
    if _moviedict_update:
        utils.write_pickle_obj(movie_pkl_file_name, movie_dict)
    if _userdict_update:
        utils.write_pickle_obj(user_pkl_file_name, user_dict)

#from experiment import *
#e = Experiment(setup_mode='lightweight')
#cf_mat = e.cf_matrix.tolil()
#u = User(0, cf_mat)
#m = Movie(0, cf_mat)


def main():
    pass

if __name__ == '__main__':
    main()

