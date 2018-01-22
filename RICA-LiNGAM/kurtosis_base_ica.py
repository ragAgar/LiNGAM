import numpy as np

class fastica():
    def __init__(self,epsilon=1e-25):
        self.epsilon      = epsilon
        
    def fit(self, X, use_sklearn=False, max_iter=1000,random_state=0,ortholization_type="seq"):
        self.n_samples, self.n_dim  = X.shape
        self.random_state = random_state
        self.max_iter       = max_iter
        
        X_center = self._centerize(X)
        self.z, self.V     = self._whitening(X_center)
        if ortholization_type == "sym":
            self._get_PDW_symO()
        elif ortholization_type == "seq":   
            self._get_PDW_seqO()
        self._unwhiten_PDW()
        return self.PDW
        
    #centerize X by X's col
    def _centerize(self, X):
        return X - np.mean(X,axis=0)

    #whitening using Eigenvalue decomposition
    def _whitening(self, X):
        E, D, E_t = np.linalg.svd(np.cov(X, rowvar=0, bias=0), full_matrices=True)
        ##変えなきゃいけない
        D = np.diag(D**(-1/2))
        V = E.dot(D).dot(E_t) #whitening matrix
        return V.dot(X.T),V
    
    #Symmetrical Ortholization
    def _get_PDW_symO(self):
        np.random.seed(self.random_state)
        W = self._sym_orthorize(np.random.uniform(low=0, size=[self.n_dim, self.n_dim]))
        
        W_t_1_copy = np.zeros(W.shape)
        W_t = np.zeros(W.shape)        
        judge = 1
        count = 0
        while np.abs(judge) > self.epsilon:
            for i in range(self.n_dim):
                w_init = W[i,:]
                W_t[i,:] = self._get_w(w_init)
            
            W_t_1 = self._sym_orthorize(W_t)
            judge = np.sum( np.dot(W_t, W_t_1) - np.eye(self.n_dim) )
            W =  W_t_1.copy()  
            count += 1
            if count > self.max_iter:
                judge = 0
            print(count)
        self.W = W_t_1
    
    #Sequential Ortholization
    def _get_PDW_seqO(self):
        np.random.seed(self.random_state)
        W_init = np.random.uniform(size=[self.n_dim, self.n_dim])  
        self.W = np.zeros(W_init.shape)
        for i in range(self.n_dim):
            w_init = W_init[i,:]
            if i > 0:
                w = self._get_w_with_orthorize(w_init, i)
                self.W[i,:] = w            
            else:
                w = self._get_w(w_init)
                self.W[i,:] = w            
            
    def _get_w(self, w):
        inner_products = []
        inner_product = 0
        count = 0
        while np.abs(inner_product - 1) > self.epsilon:
            w_t   = self._normalize(w)
            w_t_1 = self._normalize(self.z.dot(w_t.T.dot(self.z)**3) -3*w_t)

            inner_product = np.dot(w_t, w_t_1)
            w = w_t_1
            
            count += 1
            if count > self.max_iter:
                inner_product = 1
        return w_t_1

    def _get_w_with_orthorize(self, w, i):
        inner_product = 0    
        count = 0    
        while np.abs(inner_product - 1) > self.epsilon:
            w_o_t   = self._get_w(w)
            w_o_t_1 = self._orthorize(w_o_t, i)
            inner_product = np.dot(w_o_t, w_o_t_1)
            w = w_o_t_1  
            count += 1
            if count > self.max_iter:
                print("Not Converged2")
                inner_product = 1
        return w_o_t_1

    def _normalize(self, w):
        return w/np.sqrt(np.dot(w,w))     

    def _orthorize(self,w, i):
        w_add = np.zeros(w.shape)
        for j in range(i):
            w_j = self.W[j:(j+1),:].ravel()
            w_add += np.dot(w.T, w_j)*w_j
        w = w - w_add
        return self._normalize(w)
    
    def _sym_orthorize(self,W):
        E, D, E_t = np.linalg.svd(W.dot(W.T), full_matrices=True)
        ##変えなきゃいけない
        D = np.diag(D**(-1/2))
        return E.dot(D).dot(E_t).dot(W) #whitening matrix        

    #Estimate PDW
    def _unwhiten_PDW(self):
        self.PDW = self.W.dot(self.V)