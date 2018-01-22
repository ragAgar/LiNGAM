import numpy as np
import itertools
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment
from kurtosis_base_ica import fastica as k_ica
"""
Input:
    X:
        shape (n_samples, n_variables)

    use_sklearn:
        boolen value. Choose negentropy(sklearn's FastICA) or kurtosis(default).

    print_result:
        boolen value. if True, the result will be printed.
        ex.) x ---|strength|---> y
        x is the cause, y is the effect.

Output:
    the matrix of causal structure.
    x = Bx + e. B is return value

Model:
    X = BX + e
    X = Ae
    W = Xe
    z = Ve = VAe = W_ze

    X: variables
    B: Causality Matrix
    e: Exogenous variable
    z: whitening variable(check, np.cov(z) is identity matrix))


LiNGAM Estimation:
    STEP1:
        Centorize and Whitening (X) and get [z,V].
    STEP2:
        Use (z), Estimate [W_z] using kurtosis base FastICA(Independent Component Analysis).
        ※Note W_z will be estimated by each rows. Finally, Use Gram-Schmidt orthogonalization.
        ※FastICA's Estimation can't identify "The correct row's order" and "Scale"
    STEP3:
        Use (W_z, inv(V)), estimate [A,PDW].
        ※Note P is Permutation matrix, D is Diagonal matrix.
    STEP4:
        Use [PDW] and acyclicity, estimate [P,D].
        ※Note B=(I-W) and diag(B) is I because of Acyclisity.
    STEP5:
        Use [PDW,P,inv(D)], estimate [W_hat]
    STEP6:
        Prune and Permutate (W_hat) by causal order.[P_dot]
    STEP7:
        Linear Regression by causal order and replace B's value with coef.
        And get B.

"""
class LiNGAM():
    def __init__(self,epsilon=1e-25):
        self.epsilon      = epsilon

    def fit(self, X, use_sklearn=True,print_result=True,n_iter=1000,random_state=0):
        self.print_result = print_result
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_samples, self.n_dim  = X.shape
        self.X_np = self._pd2np(X)
        #return X_np

        self.PDW                = self._calc_PDW(use_sklearn)
        self.P_hat              = self._P_hat()
        self.D_hat,self.DW      = self._PW()
        self.B_hat              = self._B_hat()
        self.P_dot              = self._P_dot()
        self.B_base             = self._PBP()
        self.B_prune            = self._B_prune()
        self.B                  = self._regression_B(self.X_np)
        self.result_print()
        return self.B

    #if X is pandas DataFrame, convert numpy
    def _pd2np(self,X):
        if type(X) == pd.core.frame.DataFrame:
            X_np = np.asarray(X)
            self.columns = X.columns
        else:
            X_np = X.copy()
            self.columns = ["X%s"%(i) for i in range(self.n_dim)]
        return X_np

    """
    #whitening using Eigenvalue decomposition
    def _old_whitening(self,X):
        eigen, E = np.linalg.eig(np.cov(X, rowvar=0, bias=0))
        #eigen
        eigen[eigen<0] = -eigen[eigen<0]
        D = np.diag(eigen**(-1/2))
        V = E.dot(D).dot(E.T) #whitening matrix
        return V.dot(X.T),V
    """
    
    #Estimate W using Sklearn FastICA
    def _get_PDW_sklearn(self,X):
        ica = FastICA(max_iter=self.n_iter,random_state = self.random_state)
        S_ = ica.fit_transform(X)  # Reconstruct signals
        return ica.components_

    def _calc_PDW(self,use_sklearn):
        #use FastICA(kurtosis)
        if use_sklearn:
            PDW   = self._get_PDW_sklearn(X = self.X_np)
        #use sklearn's FastICA(neg entropy)
        else:
            PDW    = k_ica().fit(self.X_np,
                                  max_iter = self.n_iter,
                                  random_state = self.random_state)                                 
        return PDW

    #Estimate P
    def _P_hat(self):
        self.PDW[self.PDW == 0] = self.epsilon
        row_ind, col_ind = linear_sum_assignment(1/np.abs(self.PDW))
        P = np.zeros((len(row_ind),len(col_ind)))
        for i,j in zip(row_ind,col_ind):
            P[i,j] = 1
        return P

    #Estimate D and DW
    def _PW(self):
        DW = self.P_hat.dot(self.PDW)
        return np.diag(np.diag(DW)),DW

    #Estimate W and B
    def _B_hat(self):
        W_hat = np.linalg.inv(self.D_hat).dot(self.DW)
        B_hat = np.eye(len(W_hat))-W_hat
        return B_hat

    #Estimate P (permute B by causal order)
    def _P_dot(self):
        P_dot_lists = self._get_P_dot_lists()
        score = [self._calc_PBP_upper(P_dot, self.B_hat) for P_dot in P_dot_lists]
        return P_dot_lists[np.argmin(score)]
    
    #P_dot
    def _get_P_dot_lists(self):
        base_array  = np.eye(N=1,M=self.n_dim).ravel().astype("int")
        base_array  = set(itertools.permutations(base_array))
        return np.array(list(itertools.permutations(base_array)))    
    
    #get PBP to minimize upper triangle value
    def _calc_PBP_upper(self,P_dot,B_hat):
        return self._get_upper_triangle( P_dot.dot(B_hat).dot(P_dot.T)**2)
    
    #get sum of upper triangle value
    def _get_upper_triangle(self,mat):
        return np.diag(mat.dot(np.tri(self.n_dim))).sum()    

    #causal order B
    def _PBP(self):
        B_base = self.P_dot.dot(self.B_hat).dot(self.P_dot.T)
        return B_base
    
    #Prune B
    def _B_prune(self):
        #B_prune = self.P_dot.dot(self.B_hat).dot(self.P_dot.T)
        B_prune = self.B_base.copy()
        for i in range(self.n_dim):
            for j in range(i,self.n_dim):
                B_prune[i,j] = 0
        return self.P_dot.T.dot(B_prune).dot(self.P_dot)

    #Peplace B values with Regression coef
    def _regression_B(self,X):
        causal_matrix = self.B_prune.copy()
        reg_list = {i:causal_matrix[i,:] != 0 for i in range(self.n_dim)}
        for i in range(self.n_dim):
            if np.sum(reg_list[i]) != 0:
                y_reg = X[:,i]
                X_reg = X.T[reg_list[i]].T
                clf = LinearRegression()
                clf.fit(y=y_reg.reshape(self.n_samples,-1), X=X_reg.reshape(self.n_samples,-1))
                causal_matrix[i,reg_list[i]] = clf.coef_
        return causal_matrix

    #print result
    def result_print(self):
        if self.print_result:
            for i,b in enumerate(self.columns):
                for j,a in enumerate(self.columns):
                    if self.B[i,j]!=0:
                        print(a,"---|%.3f|--->"%(self.B[i,j]),b)