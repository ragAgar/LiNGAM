import numpy as np
import itertools
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment
"""
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
        self.epsilon = epsilon

    def fit(self, X, use_sklearn=False):
        X_np = self._pd2np(X)
        #return X_np
        (n_samples,self.n_dim)  = X_np.shape
        X_center   = self._centerize(X_np)
        PDW        = self._calc_PDW(X_center,use_sklearn=use_sklearn)
        P_hat      = self._P_hat(PDW)
        D_hat,DW   = self._PW(P_hat,PDW)
        B_hat      = self._B_hat(D_hat,DW)
        P_dot      = self._P_dot(B_hat)
        B_prune    = self._B_prune(P_dot,B_hat)
        return self._regression_B(X_np,B_prune,n_samples)

    #if X is pandas DataFrame, convert numpy
    def _pd2np(self,X):
        if type(X) == pd.core.frame.DataFrame:
            X_np = np.asarray(X)
        else:
            X_np = X.copy()
        return X_np

    #centerize X by X's col
    def _centerize(self,X):
        return np.asarray([X[:,i]-i_mean for i,i_mean in enumerate(np.mean(X,axis=0))]).T    

    #whitening using Eigenvalue decomposition
    def _whitening(self,X):
        eigen, E = np.linalg.eig(np.cov(X, rowvar=0, bias=0))
        D = np.diag(eigen**(-1/2))
        V = E.dot(D).dot(E.T) #whitening matrix
        return V.dot(X.T),V

    #Estimate W of Wz = s
    def _ICA(self,z,max_iter):
        dim = z.shape[0]
        W_init = np.random.uniform(size=[dim,dim])
        W = np.ones(W_init.shape)
        for i in range(dim):
            W[i,:] = self._calc_w(W_init[i,:],W,z,max_iter,i)
        return W

    #Estimate PDW
    def _PDW(self,W,V):
        A_tilde = W.T
        A = np.linalg.inv(V).dot(A_tilde)
        PDW = np.linalg.inv(A)
        return PDW

    #Estimate P
    def _P_hat(self,PDW):
        PDW[PDW == 0] = self.epsilon
        row_ind, col_ind = linear_sum_assignment(1/np.abs(PDW))
        P = np.zeros((len(row_ind),len(col_ind)))
        for i,j in  zip(row_ind,col_ind):
            P[i,j] = 1
        return P

    #Estimate D and DW
    def _PW(self,P_hat,PDW):
        DW = P_hat.dot(PDW)
        return np.diag(np.diag(DW)),DW

    #Estimate W and B
    def _B_hat(self,D_hat,DW):
        W_hat = np.linalg.inv(D_hat).dot(DW)
        B_hat = np.eye(len(W_hat))-W_hat
        return B_hat

    #Estimate P (permute B by causal order)
    def _P_dot(self,B_hat):
        n_dim = B_hat.shape[0]
        P_dot_lists = self._get_P_dot_lists()
        score = [self._calc_PBP_upper(P_dot,B_hat) for P_dot in P_dot_lists]
        return P_dot_lists[np.argmin(score)]

    #Prune B
    def _B_prune(self,P_dot,B_hat):
        B_prune = P_dot.dot(B_hat).dot(P_dot)
        for i in range(self.n_dim):
            for j in range(i,self.n_dim):
                B_prune[i,j] = 0
        return P_dot.dot(B_prune).dot(P_dot)

    #Peplace B values with Regression coef
    def _regression_B(self,X,B_prune,n_samples):
        causal_matrix = B_prune.copy()
        n_dim = causal_matrix.shape[1]
        reg_list = {i:causal_matrix[i,:] != 0 for i in range(n_dim)}
        for i in range(self.n_dim):
            if np.sum(reg_list[i]) != 0:
                y_reg = X[:,i]
                X_reg = X.T[reg_list[i]].T
                clf = LinearRegression()
                clf.fit(y=y_reg.reshape(n_samples,-1), X=X_reg.reshape(n_samples,-1))
                causal_matrix[i,reg_list[i]] = clf.coef_
        return causal_matrix

    #FastICA updates
    def _ICA_update(self,w,z):
        w = z.dot((w.T.dot(z)**3)) - 3*w
        w = w/np.sqrt(np.dot(w,w))
        return w

    #calculate w
    def _calc_w(self,w_init,W,z,max_iter,i):
        w_t_1 = w_init
        for iteration_time in range(max_iter):
            w_t = self._ICA_update(w_t_1,z)
            #w_list.append(np.abs(np.dot(w_t,w_t_1)-1))
            if (np.abs(np.dot(w_t,w_t_1)-1) < self.epsilon) or (iteration_time == (max_iter-1)):
                #without orthogonalization
                if i==0:
                    return w_t
                #orthogonalization
                else:
                    w_t = self._calc_gs(W=W,i=i)
                    if (np.abs(np.dot(w_t,w_t_1)-1) < self.epsilon) or (iteration_time == (max_iter-1)):
                        return w_t
                    else:
                        w_t_1 = w_t
            else:
                w_t_1 = w_t

    #Estimate W using Sklearn FastICA
    def _W_sklearn(self,X):
        A = FastICA(n_components=self.n_dim).fit(X).mixing_
        return np.linalg.inv(A)

    def _calc_PDW(self,X,use_sklearn):
        #use FastICA(kurtosis)
        if not use_sklearn:
            z, V   = self._whitening(X)
            W_z    = self._ICA(z,500)
            PDW    = self._PDW(W_z,V)
        #use sklearn's FastICA(neg entropy)
        else:
            PDW    = self._W_sklearn(X)
        return PDW

    #GS orthogonalization
    def _calc_gs(self,W,i):
        w_i = W[i,:]
        w_add = np.zeros(w_i.shape)
        for j in range(i):
            w_j = W[j:(j+1),:].ravel()
            w_add = w_add + np.dot(w_i,w_j)*w_j
        w_i = w_i - w_add/i
        return w_i/np.sqrt(np.dot(w_i,w_i))

    #get sum of upper triangle value
    def _get_upper_triangle(self,mat):
        return np.diag(mat.dot(np.tri(self.n_dim))).sum()

    #P_dot
    def _get_P_dot_lists(self):
        base_array  = np.eye(N=1,M=self.n_dim).ravel().astype("int")
        base_array  = set(itertools.permutations(base_array))
        return np.array(list(itertools.permutations(base_array)))

    #get PBP to minimize upper triangle value
    def _calc_PBP_upper(self,P_dot,B_hat):
        return self._get_upper_triangle( P_dot.dot(B_hat).dot(P_dot.T)**2)