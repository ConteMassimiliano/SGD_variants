import numpy as np
np.random.seed(123)

def spiderboost(X_train, y_train, w0, eta, q, K, S_len, gradient, momentum = False, beta = 1):
    """
    w0:         Starting point. Must have len equals to X_train.shape[0] 
    
    eta:        Learning rate
    
    q:          Epoch length
    
    K:          Max number of iterations.
    
    S_len:      Mini batch size.
                
    momentum:   True for apply momentum
    
    beta:       Ignored if momentum is False
    """      
    
    
    ################################
    # Spiderboost without momentum #
    ################################
    
    if not momentum:
        w = w0.copy()
        w_list = []
        n = X_train.shape[0]

        for k in range(K):
            if k % q == 0:
                v = gradient(X_train, y_train, w)
                w_list.append(w.copy())
                print("                          ", end = "\r")
                print(f"Progress {k/K * 100:.2f}%", end = "\r")
            else:
                indexes = np.random.randint(n, size = S_len)
                v += gradient(X_train[indexes, :], y_train[indexes], w) - gradient(X_train[indexes, :], y_train[indexes], w_old)

            w_old = w.copy()
            w -= eta * v


        print(" "*20, end = "\r")
        print("Done!")
        return w_list
    
    
    ################################
    # Spiderboost with momentum    #
    ################################
    
    elif momentum:
        lamb = eta
        w = w0.copy()
        u = w0.copy()
        z_list = []
        n = X_train.shape[0]

        for k in range(K):
            alpha = 2/(np.ceil(k/q) + 1)

            z = (1 - alpha) * u + alpha * w

            if k % q == 0:
                v = gradient(X_train, y_train, w)
                z_list.append(z.copy())
                print("                          ", end = "\r")
                print(f"Progress {k/K * 100:.2f}%", end = "\r")
            else:
                indexes = np.random.randint(n, size = S_len)
                v += gradient(X_train[indexes, :], y_train[indexes], z) - gradient(X_train[indexes, :], y_train[indexes], z_old)

            w_old = w.copy()
            z_old = z.copy()

            w -= lamb * v
            u = z - beta * w_old + beta * w



        print(" "*20, end = "\r")
        print("Done!")
        return z_list
    





        
        