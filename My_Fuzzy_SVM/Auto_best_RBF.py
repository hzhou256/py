#歡迎使用此方法並引用下面兩篇文章
#Li, C. H., Hsien, P. J., & Lin, L. H. (2018). A Fast and Automatic Kernel-based Classification Scheme: GDA+ SVM or KNWFE+ SVM. J. Inf. Sci. Eng., 34(1), 103-121.
#Li, C. H., Ho, H. H., Liu, Y. L., Lin, C. T., Kuo, B. C., & Taur, J. S. (2012). An Automatic Method for Selecting the Parameter of the Normalized Kernel Function to Support Vector Machines. Journal of Information Science and Engineering, 28(1), 1-15.


def JV(gv,X,y):
  from sklearn.metrics.pairwise import rbf_kernel
  import numpy as np
  K=rbf_kernel(X,gamma=gv)

  w=0
  wd=0
  for i in range(0,max(y)+1):
    idx=y==i
    WK=K[:,idx]
    WK=WK[idx,:]
    w=w+np.sum(WK)
    wd=wd+sum(idx)**2

  b=np.sum(K)-w
  bd=len(y)**2-wd

  w=w/wd
  b=b/bd

  J=1-w+b

  return J

def BestRBF(X_train,y_train):
  from scipy.optimize import minimize

  gv0=1/(X_train.shape[1]*X_train.var())
  sol=minimize(JV,gv0,args=(X_train,y_train))
  bgv=sol.x

  return bgv