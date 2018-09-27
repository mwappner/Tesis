# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:39:27 2018

@author: Marcos
"""

numero = 5

class Dummy:
    
    def __init__(self):
        self.data = list(range(numero))
        
    def cambio_data(self, num=numero):
        global numero #defino que es global
        numero = num #la modifico
        self.data = list(range(numero))