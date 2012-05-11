#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Osman Baskaya"



class Logger(object):
    
    def __init__(self, name):
        self.logFile_name = name
        #file refresh.
        f = open(self.logFile_name, 'w')
        f.close()
    

    def add(self, element, year):
        with open(self.logFile_name, 'a') as f:
            f.write(element + ' ' + year + '\n')


def main():
    pass

if __name__ == '__main__':
    main()

