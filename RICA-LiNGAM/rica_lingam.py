import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment

#from tqdm import tqdm
#from IPython.core.pylabtools import figsize

class RICA():
    def __init__(self,random_state=0, epsilon=1e-6,max_iter=1e+4):
        self.random_state = random_state
        self.epsilon = epsilon
        self.max_iter = int(max_iter)

    def fit(self,X,n_components,alpha=1,Lambda=1e+3,get_init=False):
        np.random.seed(self.random_state)
        #expect X = n*N matrix. N is number of samples
        n_dim = X.shape[1]
        #set params
        self.alpha    = alpha
        self.Lambda   = Lambda 
        self.W_shape  = (n_components, n_dim)
        self.iter_count = 0
        self.Z,self.V = self._whitening(self._centerize(X))
        self.cost_history = []
        if get_init:
            self.W_init = self._get_W_init()#np.random.randn(n_components, n_dim)
        else:
            self.W_init = np.random.randn(n_components, n_dim)
        #self.res = optimize.minimize(fun=self.cost, method="L-BFGS-B", 
        #                      x0=self.W_init.ravel(),callback=self.cbf,tol=self.epsilon)
        self.res = self.get_res_iter(self.max_iter)
        return np.linalg.inv(self.V).dot(self.res.reshape(self.W_shape).T)

    #centerize X by X's col
    def _centerize(self,X):
        return X - np.mean(X,axis=0)

    def _fastica(self):
        f = FastICA(random_state=self.random_state)
        return f.fit(self.Z.T).components_

    def _get_W_init(self):
        W_init = self._fastica()
        for i in range(self.W_shape[0]-self.W_shape[1]):
            W_init = np.vstack([W_init,np.random.randn(self.W_shape[1])])
        return W_init

    #whitening using Eigenvalue decomposition
    def _whitening(self,X):
        E, D, E_t = np.linalg.svd(np.cov(X, rowvar=0, bias=0), full_matrices=True)
        ##変えなきゃいけない
        D = np.diag(D**(-1/2))
        V = E.dot(D).dot(E_t) #whitening matrix
        return V.dot(X.T),V

    def _normalize(self,W):
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        for i in range(W.shape[0]):
            W[i,:] = W[i,:]/np.sqrt(W[i,:].dot(W[i,:]))
        return W

    def logcosh(self,x):
        return 1/self.alpha * np.log(np.cosh(self.alpha*x))

    def cost(self,W):
        W_ = W.reshape(self.W_shape)
        return self.Lambda*np.mean(np.sum((np.dot(W_.T,np.dot(W_,self.Z)) - self.Z)**2, axis=0))+\
        np.sum(np.sum(self.logcosh(W_.dot(self.Z)),axis=0))

    def cbf(self,Xi):
        self.iter_count += 1
        self.cost_history.append(self.cost(Xi))

    def get_res_iter(self,mi):
        for i in range(mi): 
            res = optimize.minimize(fun=self.cost, method="L-BFGS-B",
                                    x0=self.W_init.ravel(),callback=self.cbf,
                                    options={"maxiter":0})
            W_init = self._normalize(res.x.reshape(self.W_shape))
            if len(self.cost_history)>=3:
                if np.abs(self.cost_history[-1]-np.mean(self.cost_history[-3:])) < self.epsilon:
                    return W_init
        return W_init

class RICA_LiNGAM():
    def __init__(self,epsilon=1e-25):
        self.epsilon      = epsilon

    def fit(self, X,n_components=2, print_result=True,n_iter=1000,
            random_state=201411289, Lambda=1e+6, alpha=1,
           N_sampling=100, beta=4.3):
        #params
        self.print_result = print_result
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_samples, self.n_dim  = X.shape
        self.X_np = self._pd2np(X)
        self.beta =beta
        self.Lambda = Lambda
        self.alpha = alpha
        self.n_components = n_components
        self.N_sampling = N_sampling

        #main
        self.W_list      = self.get_W_list()
        self.prob, self.W_new = self.get_prob()
        self.B_prune = self.get_B_prune()
        self.result_print()
        return self.B_prune


    #if X is pandas DataFrame, convert numpy
    def _pd2np(self,X):
        if type(X) == pd.core.frame.DataFrame:
            X_np = np.asarray(X)
            self.columns = X.columns
        else:
            X_np = X.copy()
            self.columns = ["X%s"%(i) for i in range(self.n_dim)]
        return X_np

    def get_W_list(self):
        W_list = []
        for i in range(self.N_sampling):#tqdm()
            rica = RICA(random_state=self.random_state)
            res = rica.fit(self.get_bootstrap_X(i),n_components=self.n_components,
                           Lambda=self.Lambda, alpha=self.alpha)
            W_list.append(res)
        return np.asarray(W_list)

    def get_bootstrap_X(self,i):
        np.random.seed(i)
        n_sample = self.X_np.shape[0]
        label = np.random.choice(range(n_sample), n_sample, replace=True)
        return self.X_np[label]

    def get_prob(self):
        Prob_list = np.zeros(self.W_list.shape[1:])
        W_new = np.zeros(self.W_list.shape[1:])
        for i in range(self.W_list.shape[1]):
            for j in range(self.W_list.shape[2]):
                mu,std = np.mean(self.W_list[:,i,j]), np.std(self.W_list[:,i,j])
                Prob_list[i,j] = self.P_0(mu,std)
                W_new[i,j] = mu
        return Prob_list, W_new


    def P_0(self,mu,std):
        return np.exp(-(mu/std)**2 * 1/(2*(self.beta**2)) )

    def get_W_base(self):
        rica = RICA(random_state=self.random_state)
        return rica.fit(self.X_np,n_components=self.n_components,
                       Lambda=self.Lambda, alpha=self.alpha)


    #return tuple
    def get_use_col(self,prob):
        one_fill_prob = prob.copy()
        one_fill_prob[prob>0.5] = 1
        #check 1
        prob_sum = np.sum(prob,axis=0)
        of_prob_sum = np.sum(one_fill_prob,axis=0)
        non_use_list = []
        #もし全部0のAの列があるならば
        if np.sum(of_prob_sum == self.n_dim):
            non_use_col = np.array(range(self.n_components))[of_prob_sum == self.n_dim]
            #それが、n_comp - n_dim以上なら、確率が高いものを除去
            if len(non_use_col) >= 1:
                return tuple(np.argsort(prob_sum)[:self.n_dim])
            #それが、n_comp - n_dim未満なら、除去
            elif len(non_use_col) < 1:
                non_use_list.append(non_use_col.ravel())

        #もし片方だけ0の列があるならば
        elif np.sum(np.sum(one_fill_prob==1,axis=0)):
            #使う列を判定
            use_col = np.asarray(range(self.n_components))[np.sum(one_fill_prob==1,axis=0)==1]
            use_col = [use for use in use_col if use_col not in non_use_list]
            if len(use_col) >= self.n_dim:
                return tuple(np.asarray(use_col)[np.argsort(prob_sum[np.sum(one_fill_prob==1,axis=0) == 1])[-self.n_dim:]])
            elif len(use_col) < self.n_dim:
                return tuple(use_col)
        else:
            return tuple()

    def get_iter_list(self,prob):
        use_col = self.get_use_col(self.prob)
        if len(use_col) == 2:
            return use_col
        elif len(use_col):
            return [comb for comb in list(itertools.combinations(range(self.n_components),self.n_dim)) if self.in_check(comb,use_col)]
        else:
            return [comb for comb in list(itertools.combinations(range(self.n_components),self.n_dim))]

    def in_check(self,comb,use_col):
        check = 1
        for u in use_col:
            check = check*(u in comb)
        return check

    def get_B_prune(self):
        B_prune_list = []
        B_base_list = []
        iters = self.get_iter_list(self.prob)
        if isinstance(iters[0], tuple):
            for iter_tuple in iters:
                self.PDW = self._normalize(np.linalg.inv(self.W_new[:,iter_tuple]))
                B_prune,B_base = self.PDW2B()
                B_prune_list.append(B_prune)
                B_base_list.append(B_base)
            self.temp = B_prune_list
            B_prune = np.prod(np.asarray(B_prune_list),axis=0)
            if np.sum(B_prune) == 0:
                cost_list = []
                for i,B_base in enumerate(B_base_list):
                    cost_list.append(np.sum((B_base-B_prune_list[i])**2))
                B_prune = B_prune_list[np.argmin(cost_list)]
            return B_prune

        else:# len([iters]) == 1:
            self.PDW = self._normalize(np.linalg.inv(self.W_new[:,iters]))
            B_prune,B_base = self.PDW2B()
            return B_prune

    def PDW2B(self):
        self.P_hat              = self._P_hat()
        self.D_hat,self.DW      = self._PW()
        self.B_hat              = self._B_hat()
        self.P_dot              = self._P_dot()
        self.B_base             = self._PBP()
        return self._B_prune(), self.B_base


    def get_PDW(self):
        W_array = np.zeros((self.W_list[0,:,:].shape))
        W_P_array = np.zeros((self.W_list[0,:,:].shape))
        for i in range(self.W_list.shape[1]):
            for j in range(self.W_list.shape[2]):
                W_array[i,j]   = np.mean(self.W_list[:,i,j])
                W_P_array[i,j] = self.P_0(self.W_list[:,i,j])
        return np.linalg.inv(W_array[:,np.argsort(np.sum(W_P_array,axis=0))[-self.n_dim:]])


    def matrix_plot(self,kde,bins):
        f1 = self.W_list.shape[1]
        f2 = self.W_list.shape[2]
        #figsize(int(20*f2/(f1+f2)),int(20*f1/(f1+f2)))
        p = 0
        for i in range(f1):
            for j in range(f2):
                plt.subplot(f1*100+f2*10+1+p)
                sns.distplot(self.W_list[:,int(i%f1),j%f2],bins=bins,kde=kde)
                p += 1

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

    def _normalize(self,W):
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        for i in range(W.shape[0]):
            W[i,:] = W[i,:]/np.sqrt(W[i,:].dot(W[i,:]))
        return W

    #Prune B
    def _B_prune(self):
        #B_prune = self.P_dot.dot(self.B_hat).dot(self.P_dot.T)
        B_prune = self.B_base.copy()
        for i in range(self.n_dim):
            for j in range(i,self.n_dim):
                B_prune[i,j] = 0
        return self.P_dot.T.dot(B_prune).dot(self.P_dot)

    #print result
    def result_print(self):
        if self.print_result:
            for i,b in enumerate(self.columns):
                for j,a in enumerate(self.columns):
                    if self.B_prune[i,j]!=0:
                        print(a,"---|%.3f|--->"%(self.B_prune[i,j]),b)