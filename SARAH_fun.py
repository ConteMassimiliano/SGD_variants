import numpy as np
np.random.seed(123)

def sarah(X, y, w0, epochs, iterations_inner_loop, stepsize,b,gradients):
    
    m, n = X.shape # Number of exampleas and features
    w = w0 # Initialize weights to zero
    w_list = [w]

    for epoch in range(epochs): # iterate over epochs

        w_history = np.zeros((iterations_inner_loop,n))
        w_history[0] = w.copy()

        # v_0 = full gradient
        v = gradients(X, y, w)

        # full gradient step
        w -= v * stepsize
        w_history[1] = w.copy()

        for i in range(1,iterations_inner_loop-1):

            # random choose a sample 
            i_sample = np.random.randint(n, size = b)
            # compute accumulated direction
            v = gradients(X[i_sample,:], y[i_sample], w_history[i]) - gradients(X[i_sample,:], y[i_sample], w_history[i-1]) + v
            
            # step
            w -= v * stepsize

            w_history[i+1] = w.copy()

        # restart choosing random from m points
        t = np.random.randint(0,iterations_inner_loop)
        w = w_history[t]
        w_list.append(w)
  
    return w_list