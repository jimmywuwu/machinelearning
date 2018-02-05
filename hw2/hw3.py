import scipy.io
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import csv
import math

## hw3 
# w 39 x 1

X3_data=pd.read_csv('train.csv', sep=',',header=None)
X3_x=X3_data.iloc[:,3:]
X3_t=X3_data.iloc[:,0:3]

X3_test=pd.read_csv('test.csv', sep=',',header=None)
X3_t_x=X3_test.iloc[:,3:]
X3_t_t=X3_test.iloc[:,0:3]

def logistic_w(training_t,training_x):
	w0=np.matrix([0 for i in range(39)])
	i=0
	t=training_t.as_matrix()
	Y=activate_to_y(w0,13,training_x)
	hess_mat=[]
	gradient_out=[]
	#while(-np.matrix([([t[j][i]*math.log(Y[j][i]) for i in range(3)]) for j in range(148)]).sum()> 0.1):
	while(i<1000):
		#try:
		Y=activate_to_y(w0,13,training_x)
		gradient_out=gradient_matrix(Y,training_t,training_x)
		hess_mat=hessian_matrix(Y,training_x)
		#w0=w0-np.transpose(np.matmul(np.linalg.inv(hess_mat),np.transpose(np.matrix(gradient_out))))
		w0=w0+np.linalg.solve(hess_mat,gradient_out)
		print('%f' % -sum([sum([t[j][i]*math.log(Y[j][i]) for i in range(3)]) for j in range(148)]) )
		i=i+1
		#except:
			#print(str(i))
			#break
	print("Iteration converge at i= %d" % i )
	return {'weight':w0,'Y':Y,'hess':hess_mat,'grad':gradient_out}

# aaaa=logistic_w(X3_t,X3_x)

def activate_to_y(w,M,training_x):
	a=[[np.dot(w[0,:M],phi),np.dot(w[0,M:2*M],phi),np.dot(w[0,2*M:3*M],phi)] for phi in training_x.as_matrix()]
	return [[1/(1+math.exp(x[1]-x[0])+math.exp(x[2]-x[0])),1/(math.exp(x[0]-x[1])+1+math.exp(x[2]-x[1])),1/(math.exp(x[0]-x[2])+math.exp(x[1]-x[2])+1)] for x in a]

def gradient_matrix(y,training_t,training_x):
	t_in=training_t.as_matrix()
	x_in=training_x.as_matrix()
	grad=[]
	for j in range(3):
		tmp=[0.0 for i in range(13)]
		for n in range(148):
			tmp=tmp+(y[n][j]-t_in[n][j])*x_in[n]
		grad.extend(tmp)
	return grad


def hessian_matrix(y,training_x):
	x_in=np.matrix(training_x.as_matrix())
	#print(x_in.shape)
	hess=np.matrix([[0.0 for i in range(39)] for j in range(39)])
	tmp=np.matrix([[0.0 for i in range(13)] for j in range(13)])
	I=np.eye(3)
	for j in range(3):
		for k in range(3):
			for n in range(148):
				tmp=tmp-(y[n][k]*(I[k][j]-y[n][k])*np.matmul(np.transpose(x_in[n]),x_in[n]))

			hess[j*13:(j+1)*13,k*13:(k+1)*13]=tmp
			tmp=np.matrix([[0 for i in range(13)] for j in range(13)])

	return hess

def hessian_matrix_3(y,training_x):
	x_in=np.matrix(training_x.as_matrix())
	hess={}
	#print(x_in.shape)
	tmp=np.matrix([[0 for i in range(13)] for j in range(13)])
	for j in range(3):
		for n in range(148):
			tmp=tmp-(y[n][j]*(1-y[n][j])*np.matmul(np.transpose(x_in[n]),x_in[n]))
		hess[str(j)]=tmp
		tmp=np.matrix([[0 for i in range(13)] for j in range(13)])

	return hess

def logistic_3(training_t,training_x):
	w0=np.matrix([0 for i in range(39)])
	w1=w2=w3=np.matrix([0 for i in range(13)])
	i=0
	t=training_t.as_matrix()
	Y=activate_to_y(w0,13,training_x)
	gradient_out=[]
	hess_mat=[]
	#while(-np.matrix([([t[j][i]*math.log(Y[j][i]) for i in range(3)]) for j in range(148)]).sum()> 0.1):
	while(i<10):
		#try:
		Y=activate_to_y(w0,13,training_x)
		gradient_out=gradient_matrix(Y,training_t,training_x)
		hess_mat=hessian_matrix_3(Y,training_x)
		#w0=w0-np.transpose(np.matmul(np.linalg.inv(hess_mat),np.transpose(np.matrix(gradient_out))))
		w1=w1+np.linalg.solve(hess_mat['0'],gradient_out[:13])
		w2=w2+np.linalg.solve(hess_mat['1'],gradient_out[13:26])
		w3=w3+np.linalg.solve(hess_mat['2'],gradient_out[26:39])
		w0=np.hstack((w1,w2,w3))
		print('%f' % -sum([sum([t[j][i]*math.log(Y[j][i]) for i in range(3)]) for j in range(148)]) )
		i=i+1
		# except:
		# 	print(str(i))
		# 	break
	print("Iteration converge at i= %d" % i )
	return {'weight':w0,'Y':Y,'hess':hess_mat,'grad':gradient_out}

aaaa=logistic_3(X3_t,X3_x)


plt.hist(X3_x[,3], 50, normed=1, facecolor='g', alpha=0.75)

np.matmul(X3_t,np.array([1,2,3]))

for i in range(13):
	X3_x.iloc[:,12].groupby(np.matmul(X3_t,np.array([1,2,3]))).hist(stacked =True)
