# -*- coding: utf-8 -*-
"""
Created on Thu May 24 16:20:15 2018

@author: marcos
"""

def demo(x):
    for i in range(5):
        print("i={}, x={}".format(i, x))
        x += 1

demo(0)