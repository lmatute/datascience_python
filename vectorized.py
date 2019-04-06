#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:44:54 2019

@author: luismatute

vectorized python routines as explained in data science from scrath

These routines are not necessary as they all have been developed in Numpy
but this is all considered learing homework

"""

from functools import reduce
import math

# Vector functions

vector_names = ['vector_add','vector_sum','vector_subtract','scaler_multiply','vector_mean','vector_dot','vector_dol',
                'sum_of_squares','magnitude','squared_distance','distance','shape','get_column',
                'make_matrix','is_diagonal','rep']

def vector_add(v,w):
    """ adds corresponding elements"""
    return([v_i + w_i for v_i,w_i in zip (v,w)])
    
    
def vector_sum(vectors):
    return(reduce(vector_add,vectors))

def vector_subtract(v,w):
    """ subtracts corresponding elements"""
    return([v_i - w_i for v_i,w_i in zip (v,w)])
    
    
def scaler_multiply(a,v):
	   """ a is a number, v is a vector"""
	   return([a * v_i for v_i in v])
    
    
def vector_mean(vectors):
    """compute the vectors whose ith element  is the mean of the ith element of the input vectors"""
    n = len(vectors)
    return(scaler_multiply(1/n,vector_sum(vectors)))

def vector_dot(v,w):
    """v_1 * w_1 +....._v_n *w_n"""
    return(sum(v_i * w_i for v_i, w_i in zip(v,w)))
    
def vector_dotl(v,w):
    """v_1 * w_1 ,....._v_n *w_n"""
    return([v_i * w_i for v_i, w_i in zip(v,w)])
    
def rep(a,t):
	return([a for x in range(0,t)])
    
    
def sum_of_squares(v):
    """v_1*v_1+v_2*v_2.....v_n*v_n"""
    return(vector_dot(v,v))
    
def magnitude(v):
    return(math.sqrt(sum_of_squares(v)))

def squared_distance(v,w):
    """calculates the squared distance of two vectors"""
    return(sum_of_squares(vector_subtract(v,w)))

def distance(v,w):
    return(math.sqrt(squared_distance(v,w)))
    
# Matrix functions
    
def shape(A):
    num_rows=len(A)
    num_cols=len(A[0]) if A else 0
    return(num_rows,num_cols)
    
def get_row(A,i):
    return(A[i])
    
def get_column(A,j):
    return([A_i[j] for A_i in A])
    
def make_matrix(num_rows,num_cols,entry_fn):
    "returns a (num_rows,num_cols) matrix based on entry_fn"
    return([[entry_fn(i,j) for j in range(num_cols)] for i in range(num_rows)])

def is_diagonal(i,j):
    """ 1's on the diagonal o's every place else"""
    return(1 if i ==j else 0)

			
	

