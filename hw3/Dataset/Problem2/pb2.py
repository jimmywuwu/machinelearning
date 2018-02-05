import pandas as pd
import numpy as np 
from numpy import mat,shape,zeros
from sklearn import svm
import logging
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import sklearn.lda
Data=pd.read_csv('Iris_data.csv')

SETOSA=np.array(Data[Data["IRISTYPE Three types of iris"]=="SETOSA"])[:,:4].astype(float)
VIRGINIC=np.array(Data[Data["IRISTYPE Three types of iris"]=="VIRGINIC"])[:,:4].astype(float)
VERSICOL=np.array(Data[Data["IRISTYPE Three types of iris"]=="VERSICOL"])[:,:4].astype(float)


clf=sklearn.lda.LDA(n_components=2)  

X=np.concatenate((SETOSA,VIRGINIC,VERSICOL))
Y=np.concatenate(([[j for i in range(50)] for j in range(3)]))
clf.fit(X,Y)
pp=clf.transform(X)


clf1_X=np.concatenate((SETOSA,VIRGINIC,VERSICOL),axis=0)
clf1_Y=np.concatenate((np.array([1 for i in range(len(SETOSA))]),np.array([-1 for i in range(2*len(VIRGINIC))])),axis=0)
clf2_X=np.concatenate((VIRGINIC,SETOSA,VERSICOL),axis=0)
clf2_Y=np.concatenate((np.array([1 for i in range(len(SETOSA))]),np.array([-1 for i in range(2*len(VIRGINIC))])),axis=0)
clf3_X=np.concatenate((VERSICOL,SETOSA,VIRGINIC),axis=0)
clf3_Y=np.concatenate((np.array([1 for i in range(len(SETOSA))]),np.array([-1 for i in range(2*len(VIRGINIC))])),axis=0)
X1_trans=clf.transform(clf1_X)
X2_trans=clf.transform(clf2_X)
X3_trans=clf.transform(clf3_X)

clf1_linear=svm.SVC(kernel='linear',C=2/3,probability=True)
clf1_linear.fit(X1_trans,clf1_Y)
clf2_linear=svm.SVC(kernel='linear',C=2/3,probability=True)
clf2_linear.fit(X2_trans,clf2_Y)
clf3_linear=svm.SVC(kernel='linear',C=2/3,probability=True)
clf3_linear.fit(X3_trans,clf3_Y)

XX, YY = np.mgrid[min(X1_trans[:,0])-0.5:max(X1_trans[:,0])+0.5:200j, min(X1_trans[:,1])-0.5:max(X1_trans[:,1])+0.5:200j]
Z=[[clf3_linear.decision_function(np.matrix(i)),clf2_linear.decision_function(np.matrix(i)),clf1_linear.decision_function(np.matrix(i))] for i in  np.c_[XX.ravel(), YY.ravel()]]
Z=[[np.ravel([i for i in xx])] for xx in Z]
Z=np.array([np.argmax(xx) for xx in Z])
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z, cmap='RdBu',alpha=0.5)
plt.scatter(X1_trans[clf1_linear.support_][:,0],X1_trans[clf1_linear.support_][:,1],color='black')
plt.scatter(X2_trans[clf2_linear.support_][:,0],X2_trans[clf2_linear.support_][:,1],color='black')
plt.scatter(X3_trans[clf3_linear.support_][:,0],X3_trans[clf3_linear.support_][:,1],color='black')
plt.scatter(pp[0:50,0],pp[0:50,1],color='blue',marker='x')
plt.scatter(pp[50:100,0],pp[50:100,1],color='grey',marker='x')
plt.scatter(pp[100:150,0],pp[100:150,1],color='red',marker='x')
plt.show()


clf1_poly=svm.SVC(kernel='poly',degree=2)
clf1_poly.fit(X1_trans,clf1_Y)
clf2_poly=svm.SVC(kernel='poly',degree=2)
clf2_poly.fit(X2_trans,clf2_Y)
clf3_poly=svm.SVC(kernel='poly',degree=2)
clf3_poly.fit(X3_trans,clf3_Y)

XX, YY = np.mgrid[min(X1_trans[:,0])-0.5:max(X1_trans[:,0])+0.5:200j, min(X1_trans[:,1])-0.5:max(X1_trans[:,1])+0.5:200j]
Z=[[clf3_poly.decision_function(np.matrix(i)),clf2_poly.decision_function(np.matrix(i)),clf1_poly.decision_function(np.matrix(i))] for i in  np.c_[XX.ravel(), YY.ravel()]]
Z=[[np.ravel([i for i in xx])] for xx in Z]
Z=np.array([np.argmax(xx) for xx in Z])
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z, cmap='RdBu',alpha=0.5)
plt.scatter(X1_trans[clf1_poly.support_][:,0],X1_trans[clf1_poly.support_][:,1],color='black')
plt.scatter(X2_trans[clf2_poly.support_][:,0],X2_trans[clf2_poly.support_][:,1],color='black')
plt.scatter(X3_trans[clf3_poly.support_][:,0],X3_trans[clf3_poly.support_][:,1],color='black')
plt.scatter(pp[0:50,0],pp[0:50,1],color='blue',marker='x')
plt.scatter(pp[50:100,0],pp[50:100,1],color='grey',marker='x')
plt.scatter(pp[100:150,0],pp[100:150,1],color='red',marker='x')
plt.show()



#### svm linear #####

clf1_X=np.concatenate((SETOSA[:,0:2],VIRGINIC[:,0:2],VERSICOL[:,0:2]),axis=0)
clf1_Y=np.concatenate((np.array([1 for i in range(len(SETOSA))]),np.array([-1 for i in range(2*len(VIRGINIC))])),axis=0)
clf2_X=np.concatenate((VIRGINIC[:,0:2],SETOSA[:,0:2],VERSICOL[:,0:2]),axis=0)
clf2_Y=np.concatenate((np.array([1 for i in range(len(SETOSA))]),np.array([-1 for i in range(2*len(VIRGINIC))])),axis=0)
clf3_X=np.concatenate((VERSICOL[:,0:2],SETOSA[:,0:2],VIRGINIC[:,0:2]),axis=0)
clf3_Y=np.concatenate((np.array([1 for i in range(len(SETOSA))]),np.array([-1 for i in range(2*len(VIRGINIC))])),axis=0)

clf1_linear=svm.SVC(kernel='linear',C=2/3,probability=True)
clf1_linear.fit(clf1_X,clf1_Y)
clf2_linear=svm.SVC(kernel='linear',C=2/3,probability=True)
clf2_linear.fit(clf2_X,clf2_Y)
clf3_linear=svm.SVC(kernel='linear',C=2/3,probability=True)
clf3_linear.fit(clf3_X,clf3_Y)


XX, YY = np.mgrid[min(clf1_X[:,0])-0.5:max(clf1_X[:,0])+0.5:200j, min(clf1_X[:,1])-0.5:max(clf1_X[:,1])+0.5:200j]
Z=[[clf3_linear.decision_function(np.matrix(i)),clf2_linear.decision_function(np.matrix(i)),clf1_linear.decision_function(np.matrix(i))] for i in  np.c_[XX.ravel(), YY.ravel()]]
Z=[[np.ravel([i for i in xx])] for xx in Z]
Z=np.array([np.argmax(xx) for xx in Z])
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z, cmap='RdBu',alpha=0.5)
plt.scatter(clf1_X[clf1_linear.support_][:,0],clf1_X[clf1_linear.support_][:,1],color='black')
plt.scatter(clf2_X[clf2_linear.support_][:,0],clf2_X[clf2_linear.support_][:,1],color='black')
plt.scatter(clf3_X[clf3_linear.support_][:,0],clf3_X[clf3_linear.support_][:,1],color='black')
plt.scatter(SETOSA[:,0],SETOSA[:,1],color='blue',marker='x',s=12)
plt.scatter(VIRGINIC[:,0],VIRGINIC[:,1],color='grey',marker='x',s=12)
plt.scatter(VERSICOL[:,0],VERSICOL[:,1],color='red',marker='x',s=12)
plt.show()

#### SVM poly ####

clf1_X=np.concatenate((SETOSA[:,0:2],VIRGINIC[:,0:2],VERSICOL[:,0:2]),axis=0)
clf1_Y=np.concatenate((np.array([1 for i in range(len(SETOSA))]),np.array([-1 for i in range(2*len(VIRGINIC))])),axis=0)
clf2_X=np.concatenate((VIRGINIC[:,0:2],SETOSA[:,0:2],VERSICOL[:,0:2]),axis=0)
clf2_Y=np.concatenate((np.array([1 for i in range(len(SETOSA))]),np.array([-1 for i in range(2*len(VIRGINIC))])),axis=0)
clf3_X=np.concatenate((VERSICOL[:,0:2],SETOSA[:,0:2],VIRGINIC[:,0:2]),axis=0)
clf3_Y=np.concatenate((np.array([1 for i in range(len(SETOSA))]),np.array([-1 for i in range(2*len(VIRGINIC))])),axis=0)

clf1_poly=svm.SVC(kernel='poly',degree=2)
clf1_poly.fit(clf1_X,clf1_Y)
clf2_poly=svm.SVC(kernel='poly',degree=2)
clf2_poly.fit(clf2_X,clf2_Y)
clf3_poly=svm.SVC(kernel='poly',degree=2)
clf3_poly.fit(clf3_X,clf3_Y)

XX, YY = np.mgrid[min(clf1_X[:,0])-0.5:max(clf1_X[:,0])+0.5:200j, min(clf1_X[:,1])-0.5:max(clf1_X[:,1])+0.5:200j]
Z=[[clf3_poly.decision_function(np.matrix(i)),clf2_poly.decision_function(np.matrix(i)),clf1_poly.decision_function(np.matrix(i))] for i in  np.c_[XX.ravel(), YY.ravel()]]
Z=[[np.ravel([i for i in xx])] for xx in Z]
Z=np.array([np.argmax(xx) for xx in Z])
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z, cmap='RdBu',alpha=0.5)
plt.scatter(clf1_X[clf1_poly.support_][:,0],clf1_X[clf1_poly.support_][:,1],color='black')
plt.scatter(clf2_X[clf2_poly.support_][:,0],clf2_X[clf2_poly.support_][:,1],color='black')
plt.scatter(clf3_X[clf3_poly.support_][:,0],clf3_X[clf3_poly.support_][:,1],color='black')
plt.scatter(SETOSA[:,0],SETOSA[:,1],color='blue',marker='x',s=12)
plt.scatter(VIRGINIC[:,0],VIRGINIC[:,1],color='grey',marker='x',s=12)
plt.scatter(VERSICOL[:,0],VERSICOL[:,1],color='red',marker='x',s=12)
plt.show()


#### LDA linear ######

SETOSA_m=np.mean(SETOSA[:,:4],axis=0)
VIRGINIC_m=np.mean(VIRGINIC[:,:4],axis=0)
VERSICOL_m=np.mean(VERSICOL[:,:4],axis=0)
ALL_m=np.mean(np.concatenate((SETOSA[:,:4],VERSICOL[:,:4],VIRGINIC[:,:4])),axis=0).astype(float)
mm=np.vstack((SETOSA_m,VIRGINIC_m,VERSICOL_m,))

## Calculate LDA
S_W=np.zeros((4,4))
S_T=np.zeros((4,4))
for i in range(50):
	S_W=S_W+np.outer(SETOSA[i,0:4]-mm[0],SETOSA[i,0:4]-mm[0])+np.outer(VIRGINIC[i,0:4]-mm[1],VIRGINIC[i,0:4]-mm[1])+np.outer(VERSICOL[i,0:4]-mm[2],VERSICOL[i,0:4]-mm[2])

S_B=50*(np.outer(SETOSA_m-ALL_m,SETOSA_m-ALL_m)+np.outer(VIRGINIC_m-ALL_m,VIRGINIC_m-ALL_m)+np.outer(VERSICOL_m-ALL_m,VERSICOL_m-ALL_m))

# S_W=S_W.astype(float)
# S_B=S_B.astype(float)

A=np.linalg.eig(np.matmul(np.linalg.inv(S_W),S_B))
A=A[1][:,:2]
## Transform R4 -> R2
SETOSA_LDA=np.matmul(SETOSA,A)
VERSICOL_LDA=np.matmul(VERSICOL,A)
VIRGINIC_LDA=np.matmul(VIRGINIC,A)


clf1_linear=svm.SVC(kernel='linear',C=2/3,probability=True)
clf2_linear=svm.SVC(kernel='linear',C=2/3,probability=True)
clf3_linear=svm.SVC(kernel='linear',C=2/3,probability=True)

clf1_X=np.matmul(np.concatenate((SETOSA,VIRGINIC,VERSICOL),axis=0),A)
clf2_X=np.matmul(np.concatenate((VIRGINIC,SETOSA,VERSICOL),axis=0),A)
clf3_X=np.matmul(np.concatenate((VERSICOL,SETOSA,VIRGINIC),axis=0),A)
clf1_Y=np.concatenate((np.array([1 for i in range(len(SETOSA))]),np.array([-1 for i in range(2*len(VIRGINIC))])),axis=0)
clf2_Y=np.concatenate((np.array([1 for i in range(len(SETOSA))]),np.array([-1 for i in range(2*len(VIRGINIC))])),axis=0)
clf3_Y=np.concatenate((np.array([1 for i in range(len(SETOSA))]),np.array([-1 for i in range(2*len(VIRGINIC))])),axis=0)

clf1_linear.fit(clf1_X,clf1_Y)
clf2_linear.fit(clf2_X,clf2_Y)
clf3_linear.fit(clf3_X,clf3_Y)


XX, YY = np.mgrid[min(clf1_X[:,0])-0.02:max(clf1_X[:,0])+0.02:200j, min(clf1_X[:,1])-0.02:max(clf1_X[:,1])+0.05:200j]
Z=[[clf3_linear.decision_function(np.matrix(i)),clf2_linear.decision_function(np.matrix(i)),clf1_linear.decision_function(np.matrix(i))] for i in  np.c_[XX.ravel(), YY.ravel()]]
Z=[[np.ravel([i for i in xx])] for xx in Z]
Z=np.array([np.argmax(xx) for xx in Z])
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z, cmap='RdBu',alpha=0.5)
plt.scatter(clf1_X[clf1_linear.support_][:,0],clf1_X[clf1_linear.support_][:,1],color='black')
plt.scatter(clf2_X[clf2_linear.support_][:,0],clf2_X[clf2_linear.support_][:,1],color='black')
plt.scatter(clf3_X[clf3_linear.support_][:,0],clf3_X[clf3_linear.support_][:,1],color='black')
plt.scatter(SETOSA_LDA[:,0],SETOSA_LDA[:,1],color='blue',marker='x',s=12)
plt.scatter(VIRGINIC_LDA[:,0],VIRGINIC_LDA[:,1],color='grey',marker='x',s=12)
plt.scatter(VERSICOL_LDA[:,0],VERSICOL_LDA[:,1],color='red',marker='x',s=12)
plt.show()

######## LDA poly 2 #####


SETOSA_m=np.mean(SETOSA[:,:4],axis=0)
VIRGINIC_m=np.mean(VIRGINIC[:,:4],axis=0)
VERSICOL_m=np.mean(VERSICOL[:,:4],axis=0)
ALL_m=np.mean(np.concatenate((SETOSA[:,:4],VERSICOL[:,:4],VIRGINIC[:,:4])),axis=0).astype(float)
mm=np.vstack((SETOSA_m,VIRGINIC_m,VERSICOL_m,))

## Calculate LDA
S_W=np.zeros((4,4))
S_T=np.zeros((4,4))
for i in range(50):
	S_W=S_W+np.outer(SETOSA[i,0:4]-mm[0],SETOSA[i,0:4]-mm[0])+np.outer(VIRGINIC[i,0:4]-mm[1],VIRGINIC[i,0:4]-mm[1])+np.outer(VERSICOL[i,0:4]-mm[2],VERSICOL[i,0:4]-mm[2])

S_B=50*(np.outer(SETOSA_m-ALL_m,SETOSA_m-ALL_m)+np.outer(VIRGINIC_m-ALL_m,VIRGINIC_m-ALL_m)+np.outer(VERSICOL_m-ALL_m,VERSICOL_m-ALL_m))

# S_W=S_W.astype(float)
# S_B=S_B.astype(float)

A=np.linalg.eig(np.matmul(np.linalg.inv(S_W),S_B))
A=A[1][:,:2]
A=-4*A
## Transform R4 -> R2
SETOSA_LDA=np.matmul(SETOSA,A)
VERSICOL_LDA=np.matmul(VERSICOL,A)
VIRGINIC_LDA=np.matmul(VIRGINIC,A)


clf1_poly=svm.SVC(kernel='poly',degree=2)
clf2_poly=svm.SVC(kernel='poly',degree=2)
clf3_poly=svm.SVC(kernel='poly',degree=2)

clf1_X=np.matmul(np.concatenate((SETOSA,VIRGINIC,VERSICOL),axis=0),A)
clf2_X=np.matmul(np.concatenate((VIRGINIC,SETOSA,VERSICOL),axis=0),A)
clf3_X=np.matmul(np.concatenate((VERSICOL,SETOSA,VIRGINIC),axis=0),A)
clf1_Y=np.concatenate((np.array([1 for i in range(len(SETOSA))]),np.array([-1 for i in range(2*len(VIRGINIC))])),axis=0)
clf2_Y=np.concatenate((np.array([1 for i in range(len(SETOSA))]),np.array([-1 for i in range(2*len(VIRGINIC))])),axis=0)
clf3_Y=np.concatenate((np.array([1 for i in range(len(SETOSA))]),np.array([-1 for i in range(2*len(VIRGINIC))])),axis=0)

clf1_poly.fit(clf1_X,clf1_Y)
clf2_poly.fit(clf2_X,clf2_Y)
clf3_poly.fit(clf3_X,clf3_Y)


XX, YY = np.mgrid[min(clf1_X[:,0])-0.02:max(clf1_X[:,0])+0.02:200j, min(clf1_X[:,1])-0.02:max(clf1_X[:,1])+0.05:200j]
Z=[[clf3_poly.decision_function(np.matrix(i)),clf2_poly.decision_function(np.matrix(i)),clf1_poly.decision_function(np.matrix(i))] for i in  np.c_[XX.ravel(), YY.ravel()]]
Z=[[np.ravel([i for i in xx])] for xx in Z]
Z=np.array([np.argmax(xx) for xx in Z])
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z, cmap='RdBu',alpha=0.5)
plt.scatter(clf1_X[clf1_poly.support_][:,0],clf1_X[clf1_poly.support_][:,1],color='black')
plt.scatter(clf2_X[clf2_poly.support_][:,0],clf2_X[clf2_poly.support_][:,1],color='black')
plt.scatter(clf3_X[clf3_poly.support_][:,0],clf3_X[clf3_poly.support_][:,1],color='black')
plt.scatter(SETOSA_LDA[:,0],SETOSA_LDA[:,1],color='blue',marker='x',s=12)
plt.scatter(VIRGINIC_LDA[:,0],VIRGINIC_LDA[:,1],color='grey',marker='x',s=12)
plt.scatter(VERSICOL_LDA[:,0],VERSICOL_LDA[:,1],color='red',marker='x',s=12)
plt.show()

