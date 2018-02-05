from PIL import Image
import numpy as np
import random
import pandas as pd
import sys
from scipy.stats import multivariate_normal

k=int(sys.argv[1])
im=Image.open('hw3_img.jpg')
pixel=im.load()

Data=[]
for i in range(im.size[0]):
	for j in range(im.size[1]):
		Data.append([i,j,pixel[i,j][0],pixel[i,j][1],pixel[i,j][2]])


Data=np.array(Data)
Data= Data[:,2:]/255
def Kmeans(data,K):
	data_dim=len(data[0])
	#1 初始化 隨機決定 n個中心點, 
	m=data[random.sample(range(len(data)),K)]
	m_init=m*0
	rnk=[]
	zero=[0 for i in range(K)]
	# while(sum(sum((m-m_init)**2)<0.01)):
	for ii in range(2):
		rnk=[]
		print("hihih")
		m_init=m
		for row in data:
			#2 決定每個data 屬於哪類
			category=zero.copy()
			category[np.argmin([np.linalg.norm(row-mn) for mn in m_init])]=1
			rnk.append(category)

		#3 算每群中心
		for i in range(K):
			m[i]=0
			num=0
			for j in range(len(data)):
				m[i]=m[i]+rnk[j][i]*data[j]
				num=num+rnk[j][i]
			m[i]=m[i]/num
		
	return [m,rnk]


res=Kmeans(Data,k)
s=[0*np.outer(Data[1],Data[0]) for i in range(k)]
# 計算Kmeans 算出來的variance
for i in range(im.size[0]):
	for j in range(im.size[1]):
		cate=np.dot(res[1][i*im.size[1]+j],np.arange(0,k,1))
		s[cate]=s[cate]+np.outer((Data[i*im.size[1]+j]-res[0][cate]),(Data[i*im.size[1]+j]-res[0][cate]))

for i in range(k):
	s[i]=s[i]/sum(np.array(res[1])[:,i])

pi0=np.array([sum(np.array(res[1])[:,i])/len(Data) for i in range(k)])

def GMM(data,K,m0,s0,pi0):
	N=len(data)
	loglike=[]
	for i in range(1):
		RV=[multivariate_normal(m0[i],s0[i]) for i in range(K)]
		# E step
		rnk=np.array([normalize([pi0[i]*RV[i].pdf(row) for i in range(K)]) for row in data])
		loglike=loglike+sum([np.log(sum([pi0[ii]*RV[ii].pdf(data[jj,:]) for ii in range(K)])) for jj in range(len(data))])
		# M step
		NK=np.array([sum(rnk[:,i]) for i in range(K)])
		pi0=NK/N
		m0=np.array([sum([rnk[i][j]*data[i] for i in range(N)])/NK[j] for j in range(K)])
		s0=np.array([sum([rnk[i][j]*np.outer((data[i]-m0[j]),(data[i]-m0[j])) for i in range(N)])/NK[j] for j in range(K)])
	return [m0,rnk,loglike]



def normalize(x):
	return x/sum(x)
	
ress=GMM(Data,k,res[0],s,pi0)

print("Kmeans result with K="+str(k)+"\n")
print(res[0])
print("GMM result with K="+str(k)+"\n")
print(ress[0])

with open('GMM_like'+k+'.csv','wb') as f:
	like=ress[0][2]
	for data in like:
		f.write(data)
		f.write(',')

for i in range(im.size[0]):
	for j in range(im.size[1]):
		chh=res[0][np.dot(res[1][i*im.size[1]+j],np.arange(0,k,1))]
		pixel[i,j]=tuple([int(x) for x in chh*255])

im.save("Kmeans_"+str(k)+'.jpg')

for i in range(im.size[0]):
	for j in range(im.size[1]):
		#chh=ress[0][np.dot(ress[1][i*im.size[1]+j],np.arange(0,k,1))]
		chh=np.dot(ress[1][i*im.size[1]+j],ress[0])
		pixel[i,j]=tuple([int(x) for x in chh*255])

im.save("GMM_"+str(k)+'.jpg')

