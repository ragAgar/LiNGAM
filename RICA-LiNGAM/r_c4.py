import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
"""
input
    X: the data that you want to know causal structure.
       X.shape must be (sample, 2)
output
    The value of regression coef under estimated structure.
    if x -> y:
        y = p*x + e
    else:
        x = p*y + e
    return p

About
    1. Propose the measure that identify causality between two variables(x,y).
    2. x, y are both zero-mean.
    3. x_hat, y_hat are unit variance of x,y
    4. p is coef of regression(y = px + e)

    R_c4 = sign(kurt(x_hat))pE{x_hat^3*y_hat + x_hat*y_hat^3}
        if    R > 0 : x -> y
        elif  R < 0 : y -> x

    *if sign(kurt(x_hat)) != sign(kurt(y_hat)): fail.
"""
class R_c4():
    def __init__(self, random_state=0):
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        X = self._pd2np(X).copy()
        x     = np.asarray(X)[:,0]
        y     = np.asarray(X)[:,1]

        X_scale = self._scale(X)
        x_hat = np.asarray(X_scale)[:,0]
        y_hat = np.asarray(X_scale)[:,1]
        if np.sign(self._kurt_hat(x_hat)) != np.sign(self._kurt_hat(y_hat)):
            print("Warning!! kurt(x) and kurt(y) have defferent sign!!")
        #Coef x->y
        p_xy = self._ols_coef(x = x_hat, y = y_hat)
        R_c4  = np.sign(self._kurt_hat(x_hat)) *p_xy *np.mean(y_hat*x_hat**3 - x_hat*y_hat**3)
        #print result
        if np.sign(R_c4) == 0:
            print("Could not Estimate")
            print("%s -?- %s"%(self.columns[0],self.columns[1]))
            return 0

        #x->y
        elif R_c4 > 0:
            p_xy = self._ols_coef(x=x,y=y)
            print("%s ---|%.3f|---> %s " \
                  %(self.columns[0],p_xy,self.columns[1]))
            return p_xy

        #y->x
        elif R_c4 < 0:
            p_yx = self._ols_coef(x=y,y=x)
            print("%s <---|%.3f|--- %s " \
                  %(self.columns[0],p_yx,self.columns[1]))
            return p_yx

    def _pd2np(self,X):
        if X.shape[1] != 2:
            print("This method assume 2 varibles but got %s"%(X.shape[1]))
        if type(X) == pd.core.frame.DataFrame:
            X_np = np.asarray(X)
            self.columns = X.columns
        else:
            X_np = X.copy()
            self.columns = ["X%s"%(i) for i in range(2)]
        return X_np

    def _scale(self,X):
        return (X - np.mean(X,axis=0))/np.std(X,axis=0)

    #Note Unit Variance
    def _kurt_hat(self,x_hat):
        return np.mean(x_hat**4) - 3

    def _ols_coef(self,x,y):
        clf = LinearRegression()
        clf.fit(y=y.reshape(-1,1), X=x.reshape(-1,1))
        return clf.coef_.ravel()[0]