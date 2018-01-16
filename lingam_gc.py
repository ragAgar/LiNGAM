import numpy as np
import pandas as pd
"""
input
    X: the data that you want to know causal structure.
       X.shape must be (sample, 2)
    normalize:
        choose method. kurt-based or scaling.("kurt" or "scale")
        In paper kurt-based's score is better.
output
    None. ouly print causal structure result.

About
    -------------model---------------
        x = e_1 + beta_1*c
        y = p*x + e_2 + beta_2*c
        y_hat = p_hat*x_hat + ~~~~
    ---------------------------------

    1. Propose the measure that identify causality between two variables x,y.(can extend easily)
    2. c is Gaussian variable that is total effect of latent variables.
    3. e_1,e_2 is non-Gaussian and mutual independent.
    4. propose kurt_base normalization method:
       kurt_base normalize:
           x_hat = x/(abs(kurt(x))**(1/4))
           y_hat = y/(abs(kurt(y))**(1/4))

       *Note that, this model is under assumption |p_hat|<1.
       *That is correct assumption,
           if x_hat and y_hat are unit_variance and
               x"var(e_2) + (2*beta_1 + beta_2)*beta_2*var(c)" > 0 a
           elif sign(kurt(e_1)) == sign(kurt(e_2))

    R_xy = (Cxy + Cyx)(Cxy - Cyx)
        Cxy = C(x,y) = cum(x,x,x,y) = E{x^3*y}-3E{xy}E{x^2}
        if    R_xy > 0 : x -> y
        elif  R_xy < 0 : y -> x
"""
class LiNGAM_GC():
    def __init__(self):
        pass

    def fit(self,X, normalize="kurt"):
        X = self._pd2np_two(X)
        if normalize == "kurt":
            x,y  = self._kurt_normalization(X)
        elif normalize == "scale":
            x,y  = self._scale(X)
        else:
            print('Choose normalize="kurt" or normalize="scale"')

        C_xy = np.mean(y*x**3) - 3*np.mean(x*y)*np.mean(x**2)
        C_yx = np.mean(x*y**3) - 3*np.mean(y*x)*np.mean(y**2)
        self.R_xy = (C_xy+C_yx)*(C_xy-C_yx)
        #print(self.R_xy)
        if np.sign(self.R_xy) == 0:
            print("Could not Estimate")
            print("%s -?- %s"%(self.columns[0],self.columns[1]))
        #x->y
        elif self.R_xy > 0:
            print("%s ---> %s " %(self.columns[0],self.columns[1]))
        #y->x
        elif self.R_xy < 0:
            print("%s ---> %s " %(self.columns[1],self.columns[0]))

    def _pd2np_two(self, X):
        if X.shape[1] != 2:
            print("This method assume 2 varibles but got %s"%(X.shape[1]))
        if type(X) == pd.core.frame.DataFrame:
            X_np = np.asarray(X)
            self.columns = X.columns
        else:
            X_np = X.copy()
            self.columns = ["X%s"%(i) for i in range(2)]
        return X_np

    def _kurt_normalization(self,X):
        X = X - np.mean(X,axis=0)
        x = X[:,0]
        y = X[:,1]
        return x/(np.abs(self._kurt(x))**0.25), y/(np.abs(self._kurt(y))**0.25)

    def _scale(self,X):
        X = (X - np.mean(X,axis=0))/np.std(X,axis=0)
        return X[:,0],X[:,1]

    def _kurt(self,x):
        return np.mean(x**4) - 3*(np.mean(x**2)**2)