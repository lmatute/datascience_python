# my basic stats functions

# Pending Points for implemenattion
# dbinom(x,n,p) - see notes
# clarity when to use numpy and when to use this package



from collections import Counter
from vectorized import *
import math as m
import random
from matplotlib import pyplot as plt

stats_names =['mean','median','quantile','mode','lrange','de_mean','variance','standard_deviation',
               'interquantile_range','covariance','correlation','uniform_pdf','normal_pdf','normal_cdf',
               'inverse_normal_cdf']

def mean(x):
	return(sum(x)/len(x))
	
def median(x):
	''' function to find the median'''
	n = len(x)
	sorted_v = sorted(x)
	midpt= n //2 
	
	if n %2 == 1:
		return (sorted_v[midpt])
	else:
		lo= midpt -1
		hi =midpt
		return( (sorted_v[lo]+sorted_v[hi])/2)
		
def quantile(x,p):
	'''returns the pth percentile value in x'''
	if (p > 1):
		return(print("pls check your quantile...it's above 1"))
	p_index = int(p * len(x))
	return(sorted(x)[p_index])
	
	
def mode(x):
	"""returns a list, there may be more than one mode"""
	counts = Counter(x)
	max_count = max(counts.values())
	res = [x_i for x_i, count in counts.iteritems() if count == max_count]
	return(res)
	
	
def lrange(x):
	return(max(x)-min(x))
	
def de_mean(x):
	"""translate x by substracting its mean(so the result has mean zero)"""
	x_bar = mean(x)
	return ([x_i - x_bar for x_i in x])

def variance(x):
	"""Assumes x has at least two elements"""
	n = len(x)
	deviations = de_mean(x)
	return(sum_of_squares(deviations) /(n-1))	
	
def standard_deviation(x):
	return(math.sqrt(variance(x)))
	
def interquartile_range(x):
	return(quantile(x,.75)-quantile(x,.25))
	
def covariance(x,y):
	n = len(x)
	return(dot(de_mean(x),de_mean(y))/(n-1))
	
def correlation(x,y):
	stddev_x = standard_deviation(x)
	stddev_y = standard_deviation(y)
	if stddev_x > 0 and stddev_y > 0:
		return(covariance(x,y)/ stddev_x / stddev_y)
	else:
		return(0)
		
# Section on Distributions


def uniform_pdf(x):
	"""returns the probability that a uniform random variable is <= x"""
	if x <0: 
		return (0)
	elif (x<1) : 
		return (x)
	else: 
		return(1)
	
def normal_pdf(x, mu =0, sigma =1):
	sqrt_two_pi = math.sqrt(2 * math.pi)
	return( math.exp (-(x-mu)**2 /2 / sigma **2) / (sqrt_two_pi * sigma))

		
def normal_cdf(x,mu= 0,sigma= 1):
	return((1 + math.erf((x-mu) / (math.sqrt(2)*sigma)))/2)
	
	
def inverse_normal_cdf( p, mu = 0, sigma = 1, tolerance = 0.00001):
	""" find approximate inverse using binary search""" # if standard,compute standard and rescale 
	if mu != 0 or sigma != 1: 
		return (mu + sigma * inverse_normal_cdf( p, tolerance = tolerance))
	low_z = -10.0 # normal_cdf(-10) is (very close to) 0
	hi_z = 10.0 # normal_cdf( 10) is (very close to) 1 
	while (hi_z -low_z) > tolerance: 
		mid_z = (low_z + hi_z) / 2
		mid_p = normal_cdf( mid_z) # and the cdf's value there 
		if mid_p < p:
			low_z = mid_z
		elif mid_p > p:
			hi_z = mid_z 
		else: 
			break 
	return (mid_z)
	
def bernoulli_trial(p):
	return(1 if random.random() <p else 0 )
	
def binomial(n,p):
	return(sum(bernoulli_trial(p) for _ in range(n)))
	

# histogram of binomial trials

def make_hist(p,n,num_points):
	data = [ binomial(n,p) for _x in range(num_points)]
	# use bar chart to show the actual binomial samples
	histogram = Counter(data)
	plt.bar([x-0.4 for x in histogram.keys()],[v/num_points for v in histogram.values()],0.8,color='0.75')
	mu = p* n
	sigma = math.sqrt(n*p*(1-p))
	# use a line chart to show normal approximation
	xs = range(min(data),max(data)+1)
	ys = [normal_cdf(i+0.5,mu,sigma)-normal_cdf(i-0.5,mu,sigma) for i in xs]
	plt.plot(xs,ys)
	plt.title('Binomial Distribution vs Normal Approximation')
	plt.show()


def normal_approx_to_binomial(n,p):
	"""finds mu and sigma corresponding to a binomial(n,p)"""
	mu = p*n
	sigma = math.sqrt(p*(1-p)*n)
	return(mu,sigma)
	
normal_probability_below = normal_cdf

def normal_prob_above(lo,mu=0,sigma=1):
	return(1-normal_cdf(lo,mu,sigma))
	
def normal_prob_between(lo,hi,mu=0,sigma=1):
	return(normal_cdf(hi,mu,sigma)-normal_cdf(lo,mu,sigma))
	
def normal_prob_outside(lo,hi,mu=0,sigma=1):
	return(1-normal_prob_between(lo,hi,mu,sigma))
	
def normal_upper_bound(probability, mu =0, sigma=1):
	"""return the z for which P(Z<=z) = probability"""
	return(inverse_normal_cdf(probability,mu,sigma))
	
def normal_lower_bound(probability, mu =0, sigma=1):
	"""return the z for which P(Z>=z) = probability"""
	return(inverse_normal_cdf(1-probability,mu,sigma))
	
def normal_two_sided_bounds(probability, mu =0, sigma=1):
	"""return the symmetric (about tghe mean) bounds that contain the specified probability"""
	tail_probability = (1-probability) /2 
	upper_bound = normal_lower_bound(tail_probability,mu,sigma)
	lower_bound = normal_upper_bound(tail_probability,mu,sigma)
	return(lower_bound,upper_bound)
	

def two_sided_p_value(x,mu=0,sigma=1):
	if x> mu:
		return(2* normal_proability_above(x,mu,sigma))
	else:
		return(2*normal_probability_below(x,mu,sigma))
		
def estimated_parameters(N,n):
	# This function calcs the mean proportion and std.dev of the sampling mean proportion
	p = n/N
	sigma = math.sqrt(p*(1-p)/N)
	return(p,sigma)
	

def a_b_test_statistic(N_A,n_A,N_B,n_B):
	p_A, sigma_A = estimated_parameters(N_A,n_A)
	p_B, sigma_B = estimated_parameters(N_B,n_B)
	return(round((p_B-p_A)/math.sqrt(sigma_A ** 2 + sigma_B ** 2),2))	
	
def B(alpha, beta):
	""" a normalizingconstant so that prob is 1"""
	return(math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha+beta))

def beta_pdf(x, alpha, beta):
	if x <= 0 or x >= 1:
		return(0)
	return(x ** (alpha -1) * (1-x) ** (beta-1) / B(alpha,beta))
	
def sum_of_squares(v):
	"""Computes the sum of squares of elements in v"""
	return(sum(v_i ** 2 for v_i in v))

def difference_quotient(f,x,h):
	return((f(x+h) - f(x)) / h)
	
def square(x):
	return (x*x)
	

		
	
	
