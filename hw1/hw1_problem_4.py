import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
data_testing=pd.read_csv("4_test.csv")
data_training=pd.read_csv("4_train.csv")


def polynomail_fit(x,y,x_test,y_test,order,lam):
	x_train=pd.DataFrame([[1,i] for i in x.iloc[:,0]])
	x_test = pd.DataFrame([1,i] for i in x_test.iloc[:,0])
	if order>1:
		for j in range(2,order+1):
			x_train.insert(j,str(j),[i**j for i in x_train.iloc[:,1]])
			x_test.insert(j,str(j),[i**j for i in x_test.iloc[:,1]])
	coeficient=beta_ridge(x_train,y,lam)
	RMS=(np.dot(x_train,coeficient)-y)
	RMS=np.dot(RMS.transpose(),RMS)[0]
	RMS_t=np.dot(x_test,coeficient)-y_test
	RMS_t=np.dot(RMS_t.transpose(),RMS_t)[0]
	return {"coeficient":coeficient,"RMS":RMS/x.shape[0],"RMS_t":RMS_t/x_test.shape[0]}


def beta_ridge(x,y,lam):
	U,s,V=np.linalg.svd(x,full_matrices=1)
	a=np.zeros((x.shape[0],s.shape[0]),float)
	np.fill_diagonal(a,s/(s**2+lam))
	return np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)+np.eye(x.shape[1])*lam),x.transpose()),y)

def beta_ols(x,y):
	return np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),y)


# training the model 
# x ,y must be pd.DataFrame
x_training=pd.DataFrame(data_training.iloc[:,1])
y_training=pd.DataFrame(data_training.iloc[:,0])
x_testing = pd.DataFrame(data_testing.iloc[:,1])
y_testing=pd.DataFrame(data_testing.iloc[:,0])


data_train_draw=[]
data_test_draw=[]

data_train_draw_2=[]
data_test_draw_2=[]

for i in range(1,10):
	ans=polynomail_fit(x_training,y_training,x_testing,y_testing,i,0)
	data_train_draw.append(ans['RMS'][0])
	data_test_draw.append(ans['RMS_t'][0])
#plt.scatter([i for i in range(1,10)],data_train_draw)
plt.plot([i for i in range(1,10)],data_train_draw,label="train")
#plt.scatter([i for i in range(1,10)],data_test_draw)
plt.plot([i for i in range(1,10)],data_test_draw,label="test")
plt.legend(bbox_to_anchor=(0.85,0.8),
           bbox_transform=plt.gcf().transFigure)
plt.xlabel('Model M')
plt.ylabel('RMS')
plt.show()

for i in [-20*j/1000 for j in range(0,1001)]:
	ans2=polynomail_fit(x_training,y_training,x_testing,y_testing,9,np.exp(i))
	data_train_draw_2.append(ans2['RMS'][0])
	data_test_draw_2.append(ans2['RMS_t'][0])

#plt.scatter([i for i in range(1,10)],data_train_draw)
plt.plot([-20*j/1000 for j in range(0,1001)],data_train_draw_2,label="train")
#plt.scatter([i for i in range(1,10)],data_test_draw)
plt.plot([-20*j/1000 for j in range(0,1001)],data_test_draw_2,label="test")
plt.legend(bbox_to_anchor=(0.3,0.85),
           bbox_transform=plt.gcf().transFigure)
plt.xlabel('ln lambda')
plt.ylabel('RMS')
plt.show()
