def gradient_descent(W,y_known,y_unknown,armijo=True,alpha=1e-5,tol=1e-3,maxIter=50):
    '''
    Solves the semisupervised optimization problem using gradient descent
    with fixed stepsize alpha.
    Returns y_k,y_list,time_list where:
    y_k is the last iteration
    y_list is the list of all iterations
    time_list is the list of time required for each iteration
    Given the weight matrix W, the known labels y_known, the starting point y_unknown
    '''
    import time
    import numpy as np
    from functions import cost_function

    y_k = y_unknown.copy()
    y_list = [y_k]
    start_time = time.time()
    time_list = [0]
    
    for i in range(maxIter):
        grad = np.zeros([y_unknown.shape[0],1])

        #Gradient computation
        for j in range(len(grad)):
            k_1 = y_k[j]-y_known
            k_2 = y_k[j]-y_k
            grad[j] = 2*W[j,:].dot(np.concatenate((k_1,k_2),axis=0 ))            
 
        #Armijo rule
        if armijo:
            delta = 0.1 #each time one order of magnitude less
            gamma = 0.4
            alpha = 5
            d_k = -grad
            currest_cost = cost_function(y_k, y_known, W)
            for m in range(8):
                alpha = delta * alpha
                if cost_function(y_k + alpha * d_k, y_known, W)  < currest_cost + gamma * alpha * grad.T.dot(d_k):
                    break
                # stopping criteria
                if m == 7:
                    return y_k, y_list, time_list
                
                
        #Iteration update
        y_k = y_k-alpha*grad  
        
        y_list.append(y_k)
        time_list.append(time.time()-start_time)
        
        #Stopping criteria for non armijo
        if not armijo and np.linalg.norm(grad) < tol*len(y_unknown) : break

    return y_k, y_list, time_list

def bcgd_randomized(W,y_known,y_unknown,alpha=1e-5,maxIter=50,block_length=1):
    '''
    Solves the semisupervised optimization problem using block coordinate gradient descent
    with randomized approach and fixed stepsize alpha.
    Returns y_k,y_list,time_list where:
    y_k is the last iteration
    y_list is the list of all iterations
    time_list is the list of time required for each iteration
    Given the weight matrix W, the known labels y_known, the starting point y_unknown
    '''
    import time
    import numpy as np
    from functions import create_blocks

    y_k = y_unknown.copy()
    y_list = [y_k]
    start_time = time.time()
    time_list = [0]

    # create list of blocks
    block = create_blocks(block_length=block_length, maximum=len(y_unknown))

    for i in range(maxIter):
        
        grad = np.zeros([y_unknown.shape[0],1])
        
        #Choose one block randomly 
        block_i = np.random.randint(len(block))

        #Gradient computation in that block
        for j in block[block_i]:
            k_1 = y_k[j]-y_known
            k_2 = y_k[j]-y_k
            grad[j] = 2*W[j,:].dot(np.concatenate((k_1,k_2),axis=0))

        #Iteration update
        y_k = y_k - alpha*grad
        y_list.append(y_k)
        time_list.append(time.time()-start_time)

    return y_k,np.array(y_list),time_list

def bcgd_cyclic(W,y_known,y_unknown,alpha=1e-5,maxIter=50,block_length=1):
    '''
    Solves the semisupervised optimization problem using block coordinate gradient descent
    with cyclic approach and fixed stepsize alpha.
    Returns y_k,y_list,time_list where:
    y_k is the last iteration
    y_list is the list of all iterations
    time_list is the list of time required for each iteration
    Given the weight matrix W, the known labels y_known, the starting point y_unknown
    '''
    import time
    import numpy as np
    from functions import create_blocks

    y_k = y_unknown.copy()
    y_list = [y_k]
    start_time = time.time()
    time_list = [0]
    n_block = len(y_unknown)//block_length

    # create list of blocks
    blocks = create_blocks(block_length=block_length, maximum=len(y_unknown))
    
    for i in range(maxIter):

        # iterate over each block
        for block in blocks:

            # inizialize gradient
            grad = np.zeros([y_unknown.shape[0],1])

            # iterate inside block
            for j in block:
                k_1 = y_k[j]-y_known
                k_2 = y_k[j]-y_k
                grad[j] = 2*W[j,:].dot(np.concatenate((k_1,k_2),axis=0))

            #Iteration update
            y_k = y_k - alpha*grad

        # append iteration result
        y_list.append(y_k)
        time_list.append(time.time()-start_time)

    return y_k,np.array(y_list),time_list