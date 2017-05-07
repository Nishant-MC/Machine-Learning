#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 07:41:00 2017

@author: nm1345
"""

import numpy as np

if __name__ == "__main__":

    # Quick demo of basic setup and data members of a numpy array
    # arange can be useful for setting up monotonically increasing vector
    # reshape allows you to change the dimensions of an array
    a = np.arange(15).reshape(3, 5)
    print(a)
    print(a.shape)
    print(a.ndim)
    print(a.dtype.name)
    print(a.itemsize)
    print(a.size)
    print( type(a) )
    
    # Example of a vector (1-D array)
    # Note that you can instantiate a vector by using a sequence (or sequence of sequences, etc.)
    b = np.array([6, 7, 8])
    print(b)
    
    # You can quickly create zero vectors and matrices
    c = np.zeros(5)
    d = np.zeros((3,4))
    print(c)
    print(d)
    
    # Arrays support basic operations, and apply them elementwise
    e = np.array( [1,1,1,1,1] )
    e += 2
    e **= 2
    e -= 1
    print(e)
    f = np.array( [ 5,5,5,5,5 ] )
    print( e+f )
    
    # The dot() func can perform dot products
    # The * operator can perform multiplication b/w two vectors elementwise
    print( np.dot(e,f) )
    print ( e*f )