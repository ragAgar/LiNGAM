import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class DirectLiNGAM():
    #return standarize array of residue (y - coef*x)
    def __init__(self):
        pass

    #main
    def fit(self, X, result_print=True):
        self.x_dim = X.shape[1]
        X = self._pd2np(X)
        X_origin = X.copy()
        drop_list = []
        for i in range(self.x_dim-1):
            #drop
            #X        = np.delete(X_origin, [drop_list], axis=1)
            M_values = self._M_scores(X)
            drop_i   = np.argmax(M_values)
            drop_list.append(drop_i)
            #replace X to residue of X_i
            X        = self._reduce_X(X, drop_i)
        self.causal_order = self._true_order(drop_list,last=True)
        self.B = self._causal_effect(X_origin)
        if result_print:
            self.result_print()
        return self.causal_order

    def _pd2np(self, X):
        if type(X) == pd.core.frame.DataFrame:
            X_np = np.asarray(X)
            self.columns = X.columns
        else:
            X_np = X.copy()
            self.columns = ["X%s"%(i) for i in range(self.x_dim)]
        return X_np

    def _ols(self,x,y):
        lr = LinearRegression()
        coef_xy = lr.fit(y= y.reshape(-1, 1), X= x.reshape(-1, 1)).coef_
        coef_yx = lr.fit(y= x.reshape(-1, 1), X= y.reshape(-1, 1)).coef_
        r_xy = y - coef_xy*x
        r_yx = x - coef_yx*y
        return r_xy/np.std(r_xy), r_yx/np.std(r_yx)

    #x_j - coef*x_i and drop i
    def _reduce_X(self,X,i):
        X_new = np.zeros(X.shape)
        lr = LinearRegression()
        for j in range(X_new.shape[1]):
            lr.fit(y= X[:,j].reshape(-1, 1), X= X[:,i].reshape(-1, 1))
            X_new[:,j] = X[:,j] - lr.coef_*X[:,i]
        return np.delete(X_new, i, axis=1)

    #log cosh
    def _logcosh(self,x):
        return np.log( (np.exp(x) + np.exp(-x))/2 )

    #exp
    def _exp_func(self,x):
        return x*np.exp((-x**2)/2)

    #approximation of gaussian_entropy
    def _H_v(self):
        return (1/2)*(1+np.log(2*np.pi))

    #approximation of maximum entropy
    def _H(self,x):
        k1    = 79.047
        k2    = 7.4129
        gamma = 0.37457
        est = k1*(np.mean(self._logcosh(x))-gamma)**2 -k2*np.mean(self._exp_func(x))**2
        return self._H_v() - est

    #scaling one variable
    def _scale_x(self,x):
        x = (x - np.mean(x,axis=0))/np.std(x,axis=0)
        return x

    #convert drop_list(above) to index number.
    def _true_order(self,drop_list,last):
        drop_list = np.asarray(drop_list)
        for j in range(len(drop_list)):
            for i in range(len(drop_list)):
                is_small = drop_list[(i+1):]==drop_list[i]
                drop_list[(i+1):][is_small] = drop_list[(i+1):][is_small]+1
        if last:
            #append an index that is not found
            for i in range(self.x_dim):
                if i not in drop_list:
                    return np.append(drop_list,i)
        else:
            return drop_list

    #mutual information difference
    def _M_scores(self,X):
        #Note x_dim != self.x_dim
        x_dim = X.shape[1]
        score_list = np.zeros(x_dim)
        for i in range(x_dim):
            for j in range(i,x_dim):
                if i!=j:
                    x_i = self._scale_x(X[:,i])
                    x_j = self._scale_x(X[:,j])
                    r_ij, r_ji = self._ols(x_i, x_j)
                    m_index = (self._H(x_j) + self._H(r_ji)) - (self._H(x_i) + self._H(r_ij))
                    score_list[i] = score_list[i] + np.min([-m_index,0])**2
                    score_list[j] = score_list[j] + np.min([m_index,0])**2
        return score_list

    def _PBP(self,X):
        PBP = np.zeros([self.x_dim,self.x_dim])
        for i in range(len(self.causal_order)):
            target  = self.causal_order[i]
            explain = self.causal_order[:i]
            if len(explain):
                lr = LinearRegression()
                lr.fit(y= X[:,target].reshape(-1, 1), X= X[:,explain].reshape(-1, len(explain)))
                PBP[i,:len(explain)] = lr.coef_
        return PBP        
    
    def _P(self):
        P = np.zeros([self.x_dim,self.x_dim])
        for i,c in enumerate(self.causal_order):
            P[i,c] = 1
        return P
        
    def _causal_effect(self,X_origin):
        PBP = self._PBP(X_origin)
        P   = self._P()
        return np.dot(P.T,PBP).dot(P) 
    
    #print result
    def result_print(self):
        for i,b in enumerate(self.columns):
            for j,a in enumerate(self.columns):
                if self.B[i,j]!=0:
                    print(a,"---|%.3f|--->"%(self.B[i,j]),b)    