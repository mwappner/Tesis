# -*- coding: utf-8 -*-
"""
Created on Thu May 24 16:08:49 2018

@author: marcos
"""

def hello(name):
    """Given an object 'name', print 'Hello ' and the object."""
    print("Hello {}".format(name))


i = 42
if __name__ == "__main__":
    hello(i)