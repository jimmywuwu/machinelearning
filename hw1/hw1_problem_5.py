import numpy as np
import pandas as pd
import scipy.io

def polynomail_fit(x,y,x_test,y_test,order,lam):
    x_train=x.copy()
    x_test=x_test.copy()
    shp=x_train.shape[1]
    x_train.insert(0,"ddd",1)
    x_test.insert(0,"ddd",1)
    if (order==1):
        coeficient=beta_ridge(x_train,y,lam)
        RMS=(np.dot(x_train,coeficient)-y)
        RMS=np.dot(RMS.transpose(),RMS)[0]
        RMS_t=np.dot(x_test,coeficient)-y_test
        RMS_t=np.dot(RMS_t.transpose(),RMS_t)[0]
        return {"coeficient":coeficient,"RMS":RMS/x_train.shape[0],"RMS_t":RMS_t/x_test.shape[0]}
    elif(order==2):
        i=shp
        for x in range(1,shp+1):
            for yy in range(x,shp+1):
                x_train.insert(i,"x"+str(x)+"x"+str(yy),x_train.iloc[:,x]*x_train.iloc[:,yy])
                x_test.insert(i,"x"+str(x)+"x"+str(yy),x_test.iloc[:,x]*x_test.iloc[:,yy])
                i=i+1
        coeficient=beta_ridge(x_train,y,lam)
        RMS=(np.dot(x_train,coeficient)-y)
        RMS=np.dot(RMS.transpose(),RMS)[0]
        RMS_t=np.dot(x_test,coeficient)-y_test
        RMS_t=np.dot(RMS_t.transpose(),RMS_t)[0]
        return {"coeficient":coeficient,"RMS":RMS/x_train.shape[0],"RMS_t":RMS_t/x_test.shape[0]}

def beta_ridge(x,y,lam):
    U,s,V=np.linalg.svd(x,full_matrices=1)
    a=np.zeros((x.shape[0],s.shape[0]),float)
    np.fill_diagonal(a,s/(s**2+lam))
    return np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)+np.eye(x.shape[1])*lam),x.transpose()),y)

Y=scipy.io.loadmat("5_T.mat")
X=scipy.io.loadmat("5_X.mat")

X=pd.DataFrame(X['X'])
Y=pd.DataFrame(Y['T'])

x_class1_train=X.iloc[0:40,:]
x_class2_train=X.iloc[50:90,:]
x_class3_train=X.iloc[100:140,:]
x_class1_test=X.iloc[40:50,:]
x_class2_test=X.iloc[90:100,:]
x_class3_test=X.iloc[140:150,:]

y_class1_train=Y.iloc[0:40,:]
y_class2_train=Y.iloc[50:90,:]
y_class3_train=Y.iloc[100:140,:]
y_class1_test=Y.iloc[40:50,:]
y_class2_test=Y.iloc[90:100,:]
y_class3_test=Y.iloc[140:150,:]

X_train=x_class1_train.append(x_class2_train).append(x_class3_train)
Y_train=y_class1_train.append(y_class2_train).append(y_class3_train)

X_test=x_class1_test.append(x_class2_test).append(x_class3_test)
Y_test=y_class1_test.append(y_class2_test).append(y_class3_test)

polynomail_fit(X_train,Y_train,X_test,Y_test,2,0)["RMS"]
polynomail_fit(X_train.iloc[:,[0,1,2]],Y_train,X_test.iloc[:,[0,1,2]],Y_test,2,0)["RMS_t"]
polynomail_fit(X_train.iloc[:,[0,1,3]],Y_train,X_test.iloc[:,[0,1,3]],Y_test,2,0)["RMS_t"]
polynomail_fit(X_train.iloc[:,[1,2,3]],Y_train,X_test.iloc[:,[1,2,3]],Y_test,2,0)["RMS_t"]
polynomail_fit(X_train.iloc[:,[0,2,3]],Y_train,X_test.iloc[:,[0,2,3]],Y_test,2,0)["RMS_t"]



