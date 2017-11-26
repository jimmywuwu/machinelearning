import scipy.io
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import csv
import math

X2=scipy.io.loadmat("2_data.mat")
X2_x=X2['x']
X2_x=[i[0] for i in X2_x]
X2_t=X2['t']
X2_t=[i[0] for i in X2_t]

def hw2(x2_x,x2_t,S0,m0,n,beta):
	Design_matrix=np.matrix([[sigmoid((item-2*j/7)/0.1) for j in range(0,7)] for item in x2_x])
	S_n=np.linalg.inv(np.linalg.inv(S0)+beta*np.matmul(np.transpose(Design_matrix[0:n]),Design_matrix[0:n]))
	Miu_n=np.matmul(S_n,np.matmul(np.linalg.inv(S0),np.transpose(m0))+beta*np.matmul(np.transpose(Design_matrix[0:n]),np.transpose(np.matrix(x2_t[0:n]))))
	return  {'mu_n':Miu_n,'s_n':S_n}



draw1=hw2(X2_x,X2_t,(10**6)*np.eye(7),np.matrix([0,0,0,0,0,0,0]),10,1)
draw2=hw2(X2_x,X2_t,(10**6)*np.eye(7),np.matrix([0,0,0,0,0,0,0]),15,1)
draw3=hw2(X2_x,X2_t,(10**6)*np.eye(7),np.matrix([0,0,0,0,0,0,0]),30,1)
draw4=hw2(X2_x,X2_t,(10**6)*np.eye(7),np.matrix([0,0,0,0,0,0,0]),80,1)

sample_1=np.random.multivariate_normal([float(i) for i in draw4['mu_n']], draw4['s_n'], size=5)

x_draw=[[float(mean_post(x/500,sample_1[i])) for x in range(1001)] for i in range(5)]

for i in x_draw:
	plt.plot([x/500 for x in range(1001)],i)

for i in range(10):
	plt.scatter(X2_x[i],X2_t[i])


def draw_predict_2(n):
	draww=hw2(X2_x,X2_t,(10**6)*np.eye(7),np.matrix([0,0,0,0,0,0,0]),n,1)
	xx_draw=[[x/500,float(mean_post(x/500,draww['mu_n'])-sig_post(x/500,draww['s_n'],1)),float(mean_post(x/500,draww['mu_n'])),float(mean_post(x/500,draww['mu_n'])+sig_post(x/500,draww['s_n'],1))] for x in range(1000)]
	x=[i[0] for i in xx_draw]
	y1=[i[1] for i in xx_draw]
	y2=[i[2] for i in xx_draw]
	y3=[i[3] for i in xx_draw]
	fig,ax=plt.subplots(1,1,sharex=True)
	ax.plot(x,y2,color=(1,0.5,0.5))
	ax.fill_between(x, y1, y3,facecolor=(1, 0.5, 0.5,0.5))
	for i in range(n):
		ax.scatter(X2_x[i],X2_t[i],s=30,color=(0.5,0.5,0.5))






def sigmoid(a):
	return 1/(1+np.exp(-a))

def mean_post(x,m_n):
	return np.dot(np.transpose(m_n),np.transpose(np.matrix([sigmoid((x-2*j/7)/0.1) for j in range(0,7)])))
	

def sig_post(x,S_n,beta):
	phi=np.matrix([sigmoid((x-2*j/7)/0.1) for j in range(0,7)])
	return 1/beta+np.dot(np.dot(phi,S_n),np.transpose(phi))
