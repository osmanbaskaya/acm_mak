#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Osman Baskaya"

import re
from mediawikifetcher import PATH
from os import listdir


class Article(object):
    
    def __init__(self, title, text, year='0'):
        if '=' in title:
            title = title.replace('=', '/')
        self.title = title
        self.year = year
        self.text = text

    def clean_all_references(self):
        self.text = re.sub(r'\[\d*\]', '', self.text)

    def clean_word(self, regEx):
        pass

    def __repr__(self):
        return self.title

def get_all_offline_articles(folder='movies'):
    all_article_name = listdir(PATH + folder)
    all_article = dict()
    for article_name in all_article_name:
        with open(PATH + folder + "/" + article_name) as f:
            text = ''.join(f.readlines())
            title = article_name[0:len(article_name)-4]
            title = title.lower()
            article = Article(title=title, text=text)
            article.clean_all_references()
            all_article[title] = article
    return all_article



def main():
    pass

if __name__ == '__main__':
    main()

