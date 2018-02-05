import scipy.io
import numpy as np
from numpy import matmul
import matplotlib as plt

data=scipy.io.loadmat("2_data.mat")
t=data['t']
x=data['x']

t=[i[0] for i in t]
x=[i[0] for i in x]

Train_t=t[0:60]
Train_x=x[0:60]

Test_t=t[60:]
Test_x=x[60:]


def pb1_1(test_x,test_t,train_x,train_t,theta_0,theta_1,theta_2,theta_3):
	C_N=[[kernel(x,y,theta_0,theta_1,theta_2,theta_3) for x in train_x] for y in train_x]
	for i in range(len(train_x)):
		C_N[i][i]=C_N[i][i]+1
	K=[[kernel(x,y,theta_0,theta_1,theta_2,theta_3) for x in np.arange(0,2,0.01)] for y in train_x]
	K_train=[[kernel(x,y,theta_0,theta_1,theta_2,theta_3) for x in train_x] for y in train_x]
	K_test=[[kernel(x,y,theta_0,theta_1,theta_2,theta_3) for x in test_x] for y in train_x]
	C_N_inv=np.linalg.inv(np.matrix(C_N))
	K=np.matrix(K)
	m=np.matmul(np.matmul(np.transpose(K),C_N_inv),np.transpose(np.matrix(train_t)))
	m_train=np.matmul(np.matmul(np.transpose(K_train),C_N_inv),np.transpose(np.matrix(train_t)))
	m_test=np.matmul(np.matmul(np.transpose(K_test),C_N_inv),np.transpose(np.matrix(train_t)))
	A=np.matmul(np.matmul(np.transpose(K),C_N_inv),K)
	s=[kernel(x,x,theta_0,theta_1,theta_2,theta_3)+1-A[int(100*x),int(100*x)] for x in np.arange(0,2,0.01)]
	# 將資料處理成 [200,] 畫圖比較不會有 dimension 的問題
	s=np.asarray(s)
	m=np.squeeze(np.asarray(m))
	m_train=np.squeeze(np.asarray(m_train))
	m_test=np.squeeze(np.asarray(m_test))
	plt.plot(np.arange(0,2,0.01),m,color='red')
	plt.fill_between(np.arange(0,2,0.01),np.transpose(m)-s,np.transpose(m)+s,facecolor=(1, 0.5, 0.5,0.5))
	plt.scatter(train_x,train_t,s=3,color='red')

	rms_train=sum((m_train-np.array(train_t))**2)/len(train_t)
	rms_test=sum((m_test-np.array(test_t))**2)/len(test_t)

	return {'train':rms_train,'test':rms_test}
	
pb1_1(Test_x,Test_t,Train_x,Train_t,1,4,0,0)

def kernel(x1,x2,theta_0,theta_1,theta_2,theta_3):
	return theta_0*np.exp(-theta_1/2*(x1-x2)**2)+theta_2+theta_3*x1*x2

def C_theta0(x1,x2,theta_0,theta_1,theta_2,theta_3):
	return np.exp(-1/2*theta_1*(x1-x2)**2)

def C_theta1(x1,x2,theta_0,theta_1,theta_2,theta_3):
	return (-1/2*(x1-x2)**2)*theta_0*np.exp(-1/2*theta_1*(x1-x2)**2)

def C_theta2(x1,x2,theta_0,theta_1,theta_2,theta_3):
	return 1

def C_theta3(x1,x2,theta_0,theta_1,theta_2,theta_3):
	return x1*x2

pb1_1(Test_x,Test_t,Train_x,Train_t,1,4,0,0)
pb1_1(Test_x,Test_t,Train_x,Train_t,0,0,0,1)
pb1_1(Test_x,Test_t,Train_x,Train_t,1,4,0,5)
pb1_1(Test_x,Test_t,Train_x,Train_t,1,64,10,0)

def pb1_2(test_x,test_t,train_x,train_t,theta_0,theta_1,theta_2,theta_3):
	j=1
	update0=10
	update1=10
	update2=10
	update3=10
	THETA=[[theta_0,theta_1,theta_2,theta_3]]
	while((update0>=6)|(update1>=6)|(update2>=6)|(update3>=6)):
		C_N=[[kernel(x,y,theta_0,theta_1,theta_2,theta_3) for x in train_x] for y in train_x]
		for i in range(len(train_x)):
			C_N[i][i]=C_N[i][i]+1
		C_N_inv=np.linalg.inv(np.matrix(C_N))
		C_theta_0=np.matrix([[C_theta0(x,y,theta_0,theta_1,theta_2,theta_3) for x in train_x] for y in train_x])
		C_theta_1=np.matrix([[C_theta1(x,y,theta_0,theta_1,theta_2,theta_3) for x in train_x] for y in train_x])
		C_theta_2=np.matrix([[C_theta2(x,y,theta_0,theta_1,theta_2,theta_3) for x in train_x] for y in train_x])
		C_theta_3=np.matrix([[C_theta3(x,y,theta_0,theta_1,theta_2,theta_3) for x in train_x] for y in train_x])
		A=np.matmul(np.matrix(train_t),C_N_inv)
		update0=float(-1/2*np.trace(np.matmul(C_N_inv,C_theta_0))+1/2*np.matmul(np.matmul(A,C_theta_0),np.transpose(A)))
		update1=float(-1/2*np.trace(np.matmul(C_N_inv,C_theta_1))+1/2*np.matmul(np.matmul(A,C_theta_1),np.transpose(A)))
		update2=float(-1/2*np.trace(np.matmul(C_N_inv,C_theta_2))+1/2*np.matmul(np.matmul(A,C_theta_2),np.transpose(A)))
		update3=float(-1/2*np.trace(np.matmul(C_N_inv,C_theta_3))+1/2*np.matmul(np.matmul(A,C_theta_3),np.transpose(A)))
		theta_0=theta_0+0.001*update0
		theta_1=theta_1+0.001*update1
		theta_2=theta_2+0.001*update2
		theta_3=theta_3+0.001*update3
		print(theta_0,theta_1,theta_2,theta_3)
		THETA.append([theta_0,theta_1,theta_2,theta_3])
		j=j+1
	return [theta_0,theta_1,theta_2,theta_3]

def Baysain_Linear_Regression(x2_x,x2_t,S0,m0,n,beta):
	Design_matrix=np.matrix([[sigmoid((item-2*j/7)/0.1) for j in range(0,7)] for item in x2_x])
	S_n=np.linalg.inv(np.linalg.inv(S0)+beta*np.matmul(np.transpose(Design_matrix[0:n]),Design_matrix[0:n]))
	Miu_n=np.matmul(S_n,np.matmul(np.linalg.inv(S0),np.transpose(m0))+beta*np.matmul(np.transpose(Design_matrix[0:n]),np.transpose(np.matrix(x2_t[0:n]))))
	return  {'mu_n':Miu_n,'s_n':S_n}
def sigmoid(a):
	return 1/(1+np.exp(-a))

def mean_post(x,m_n):
	return np.dot(np.transpose(m_n),np.transpose(np.matrix([sigmoid((x-2*j/7)/0.1) for j in range(0,7)])))
	

def sig_post(x,S_n,beta):
	phi=np.matrix([sigmoid((x-2*j/7)/0.1) for j in range(0,7)])
	return 1/beta+np.dot(np.dot(phi,S_n),np.transpose(phi))

def draw_predict_2(n):
	draww=Baysain_Linear_Regression(Train_x,Train_t,(10**6)*np.eye(7),np.matrix([0,0,0,0,0,0,0]),n,1)
	xx_draw=[[x/500,float(mean_post(x/500,draww['mu_n'])-sig_post(x/500,draww['s_n'],1)),float(mean_post(x/500,draww['mu_n'])),float(mean_post(x/500,draww['mu_n'])+sig_post(x/500,draww['s_n'],1))] for x in range(1000)]
	x=[i[0] for i in xx_draw]
	y1=[i[1] for i in xx_draw]
	y2=[i[2] for i in xx_draw]
	y3=[i[3] for i in xx_draw]
	fig,ax=plt.subplots(1,1,sharex=True)
	ax.plot(x,y2,color=(1,0.5,0.5))
	ax.fill_between(x, y1, y3,facecolor=(1, 0.5, 0.5,0.5))
	ax.scatter(Train_x,Train_t,s=2.5,color=(0.5,0.5,0.5))

draw_predict_2(60)


res=pb1_2(Test_x,Test_t,Train_x,Train_t,3,6,4,5)

pb1_1(Test_x,Test_t,Train_x,Train_t,res[0],res[1],res[2],res[3])





