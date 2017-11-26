import scipy.io
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import csv
import math
#1 number 1 

X1=scipy.io.loadmat("1_data.mat")
X1=X1["r2"]

def hw1(x1,n):
	lambda_map=np.matrix([[0,0],[0,0]])
	numer=0
	v0=1
	p=2
	for item in x1:
		if(numer<n):
			lambda_map=lambda_map+np.matmul(np.transpose(np.matrix(item-[1,-1])),np.matrix(item-[1,-1]))
		numer=numer+1

	#np.linalg.inv(lambda_map+np.matrix([[1,0],[0,1]]))/(n+v0-p+1)
	return np.linalg.inv(lambda_map+np.matrix([[1,0],[0,1]]))/(n+v0-p+1)
