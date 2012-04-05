#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Osman Baskaya"

import wikipydia as wp
from bs4 import BeautifulSoup
import codecs
from logger import Logger
from time import sleep
from sys import stderr
from os import listdir
from datetime import datetime

PATH = '/home/tyr/github/acm_mak/'
DATABASE = PATH + '1m/movies.dat'
DELIMITER = '::'
INDEX = 1
DELAY = 2

#colors
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

def get_soup(page):
    soup = BeautifulSoup(page['html'])
    return soup

def get_all_parag(soup):
    all_parag = soup.find_all('p')
    text = ''
    for parag in all_parag:
        text += parag.get_text() + '\n'
    #return ''.join(all_parag)
    return text


def writeMovie2File(movie_name, soup):
    if '/' in movie_name:
        movie_name = movie_name.replace('/', '=')
    f = codecs.open(PATH + 'movies/' + movie_name + '.txt', "w", "utf-8")
    f.write(get_all_parag(soup))
    g = codecs.open(PATH + 'source/' + movie_name + '.html', "w", "utf-8")
    g.write(soup.text)

    f.close()
    g.close()
    print '%s has successfully written' % movie_name


def read_movie_database(filename=DATABASE, delimiter=DELIMITER, index=INDEX):
    movies = []
    with open(filename) as f:
        for line in f.readlines():
            movie = line.split(delimiter)[index]
            movies.append(movie)
    return movies

def get_search_format(movie):
    w = movie.split('(')
    name = w[0]
    year = w[-1]
    return name[:len(name)-1], year[:len(year)-1]

def search_key_combination(movie, year):
    key = '%s (%s film)'
    res = connect_query(key % (movie, year), query = 'search') 
    #res = wp.opensearch(key % (movie, year))
    #sleep(DELAY)
    if not res[1]:
        key = '%s (film)'
        #res = wp.opensearch(key % movie)
        res = connect_query(key % movie, query = 'search')
        #sleep(DELAY)
        if not res[1]:
            res = connect_query(movie, query = 'search')
            #res = wp.opensearch(movie)
    return res[1]

def is_correct_page(soup, year):
    text = ''
    try:
        text = soup.p.get_text()
    except AttributeError:
        err_text = COLOR_SEQ % (30 + RED) + "Attribute Error in is_correct_page" \
                              + RESET_SEQ
        stderr.write(err_text)
    return (('film' in text or 'directed' in text or 'movie' in text) \
            and (year in text or str(int(year)-1) in text or str(int(year)+1)))

def get_download_list(folder_name='movies', path = PATH):
    down_list = listdir(PATH + folder_name)
    formatted_list = [] 
    for element in down_list:
        element = element.split('.txt')
        if '=' in element[0]:
            print element
            element[0] = element[0].replace('=', '/')
        formatted_list.append(element[0])

    return formatted_list

def get_inexisted_movie_list(logfile_name=None):
    movies = []
    if logfile_name is None:
        pass # get the last one
    
    with open(PATH + 'logs/' + logfile_name) as f:
        for line in f.readlines():
            element = line.rsplit(' ', 1)
            movies.append(element[0])

    return movies


def get_correct_article(movie, year):

    article_found = False
    soup = None

    for i in range(2):
        art_namelist = search_key_combination(movie, year)
        if art_namelist:
            for article_name in art_namelist:
                page = connect_query(article_name, query='render')
                if page is not None:
                    soup = get_soup(page)
                    if is_correct_page(soup, year):
                        article_found = True
                        break
        if (article_found is False) and (',' in movie):
            splitted_movie = movie.rsplit(',', 1)
            movie = ' '.join(reversed(splitted_movie))
        else:
            break

    return soup if article_found else None
    
    

def get_all_articles(movies, logger=None, download_list=[]):
    
    if logger is None:
        time = str(datetime.now())
        logger = Logger(PATH + 'logs/' + time)

    for movie in movies:
        movie, year = get_search_format(movie)
        if movie not in download_list:
            soup = get_correct_article(movie, year)
            if soup is not None:
                writeMovie2File(movie, soup)
            else:
                logger.add(movie, year)
                not_found_inf = COLOR_SEQ % (30 + YELLOW) + "%s couldn't find" \
                              + RESET_SEQ
                print not_found_inf % movie
        else:
            infor = COLOR_SEQ % (30 + GREEN) + "%s already downloaded" \
                                  + RESET_SEQ
            print infor % movie

def connect_query(*args, **kwargs):
    res = None
    func = None

    if kwargs['query'] == 'search':
        func = wp.opensearch
    elif kwargs['query'] == 'render':
        func = wp.query_text_rendered
    else:
        raise "Unrecognized Query"
        exit(1)

    counter = 0
    CONNECTION_LIMIT = 30 
    while (res is None):
        counter += 1
        if CONNECTION_LIMIT <= counter:
            print 'Problem is bigger than you except. Check your Internet, Sir!'
            exit(1)
        try:
            res = func(*args)
            sleep(DELAY)
        except IOError as e:
            print "Connection Problem: Trying to connect again after 5 sec"
            print e.message
            sleep(5)
        except KeyError as e:
            print "Wikipydia problems"
    return res



def main():
    #print read_movie_database()
    movies = set(read_movie_database())
    download_list = get_download_list()
    inexisted_movie_list = get_inexisted_movie_list('alog')

    download_list.extend(inexisted_movie_list)
    
    # Preparing the Logger
    time = str(datetime.now())
    logger = Logger(PATH + 'logs/' + time)


    get_all_articles(movies, logger, download_list)


if __name__ == '__main__':
    main()

