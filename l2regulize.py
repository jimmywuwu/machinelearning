import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
## Generate Sample

y=pd.DataFrame([i/100+np.random.normal(0,1) for i in range(101)])
x=pd.DataFrame([[1,i/100] for i in range(101)])
x=pd.DataFrame([[1,i/100,(i/100)**2] for i in range(101)])
x=pd.DataFrame([[1,i/100,(i/100)**2,(i/100)**3,(i/100)**4,(i/100)**5,(i/100)**6] for i in range(101)])

stats.mstats.lin

plt.scatter(x,y)
plt.ylabel('some numbers')
plt.show()
## calculate l2 regulization
def beta_ridge(x,y,lam):
	U,s,V=np.linalg.svd(x,full_matrices=1)
	a=np.zeros((x.shape[0],s.shape[0]),float)
	np.fill_diagonal(a,s/(s**2+lam))
	return np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)+np.eye(x.shape[1])*lam),x.transpose()),y)
## Cross validation to tune "lambda"
def choose_best_lambda(x,y,n):
	MSE=[]
	for lam in [j/100 for j in range(n)]:
		MSE_tmp=0
		for i in range(x.shape[0]-1):
			tmp_x= x[0:i].append(x[(i+1):x.shape[0]])
			tmp_y= y[0:i].append(y[(i+1):x.shape[0]])
			MSE_tmp=MSE_tmp+(float)(np.dot(beta_ridge(tmp_x,tmp_y,lam).transpose(),x.iloc[i,])-tmp_y.iloc[i,])**2
		MSE.append(MSE_tmp)
	return np.argmin(MSE)/100


## SVD to calculate
order=[]
for i in range(101):

i=10
clif=RidgeCV(alphas=[i/100 for i in range(1001)])
clif.fit(x,y)
clif.coef_
mean_squared_error(clf.predict(x),y)
order.append(mean_squared_error(clf.predict(x),y))

np.argmin(order)/100

