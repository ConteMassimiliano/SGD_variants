import numpy as np

def katyusha_minibatch_noprox(X_train,y_train,gradient_fun,x0,S,L,b,option = 1,t2=0.5):
    n = X_train.shape[0]
    m = int(np.ceil(2*n/b))
    #t2 = 0.5
    x_snap = y = z = x = x0
    x_list = []
    for s in range(S):
        t1 = 2/(s+4)
        alpha = 1/(3*t1*L)
        mu = gradient_fun(X_train,y_train,x_snap)
        x_snap2 = np.zeros(x_snap.shape)
        for k in range(m):
            x = t1*z+t2*x_snap+(1-t1-t2)*y
            grad = mu.copy()
            indexes = np.random.randint(n, size = b)
            grad += (gradient_fun(X_train[indexes,:],y_train[indexes],x) - gradient_fun(X_train[indexes,:],y_train[indexes],x_snap))
            z_old = z.copy()
            z= z - alpha*grad
            if option == 1:
                y = x-grad*(1/(3*L))
            else:
                y = x+t1*(z-z_old)
                #y =  y - t1*(z-z_old)
            x_snap2 += (1/m)*y
        x_snap = x_snap2
        x_list.append(x_snap)
    return x_list

def katyusha_minibatch(X_train,y_train,gradient_fun,proximal_minimizer,x0,S,L,b,option = 1,t2=0.5):
    n = X_train.shape[0]
    m = int(np.ceil(2*n/b))
    #t2 = 0.5
    x_snap = y = z = x = x0
    x_list = []
    for s in range(S):
        t1 = 2/(s+4)
        alpha = 1/(3*t1*L)
        mu = gradient_fun(X_train,y_train,x_snap)
        x_snap2 = np.zeros(x_snap.shape)
        for k in range(m):
            x = t1*z+t2*x_snap+(1-t1-t2)*y
            grad = mu.copy()
            indexes = np.random.randint(n, size = b)
            grad += (gradient_fun(X_train[indexes,:],y_train[indexes],x) - gradient_fun(X_train[indexes,:],y_train[indexes],x_snap))
            z_old = z.copy()
            z= proximal_minimizer(z,grad,alpha)
            if option == 1:
                y = proximal_minimizer(x,grad,(1/(3*L)))
            else:
                y = x+t1*(z-z_old)
                #y =  y - t1*(z-z_old)
            x_snap2 += (1/m)*y
        x_snap = x_snap2
        x_list.append(x_snap)
    return x_list


def proximal_minimizer_lr_nc(zk,grad,alpha,gamma,maxIter = 10):
    #def g(z):
    #    return 1/(2*alpha)*np.linalg.norm(z-zk)**2+np.dot(grad,z)+gamma*np.sum(z**2/(1+z**2))
    z = zk-alpha*grad
    #cost_list = [g(z)]
    for i in range(maxIter):
        z = zk-alpha*grad-alpha*gamma*(2*z)/(1+z**2)**2
        #cost_list.append(g(z))
    #return z,cost_list
    return z

def proximal_minimizer_lr_nc_gd(zk,grad,alpha,gamma,maxIter = 10,stepsize = 0.1):
    #def g(z):
    #    return 1/(2*alpha)*np.linalg.norm(z-zk)**2+np.dot(grad,z)+gamma*np.sum(z**2/(1+z**2))
    z = zk-alpha*grad
    #cost_list = [g(z)]
    for i in range(maxIter):
        z = z-stepsize*( (1/alpha)*(z-zk)+grad+gamma*(2*z)/(1+z**2)**2)
        #cost_list.append(g(z))
    #return z,cost_list
    return z

def proximal_minimizer_lr_convex_reg(zk,grad,alpha,gamma,maxIter = 10):
    #def g(z):
    #    return 1/(2*alpha)*np.linalg.norm(z-zk)**2+np.dot(grad,z)+gamma*np.linalg.norm(z)**2
    z = zk-alpha*grad
    #cost_list = [g(z)]
    for i in range(maxIter):
        z = zk-alpha*grad-alpha*gamma*(2*z)
        #cost_list.append(g(z))
    return z

def proximal_minimizer_rr_l1(zk,grad,alpha,gamma,maxIter = 10):
    #def g(z):
    #    return 1/(2*alpha)*np.linalg.norm(z-zk)**2+np.dot(grad,z)+gamma*np.linalg.norm(z)**2
    z = zk-alpha*grad
    #cost_list = [g(z)]
    for i in range(maxIter):
        z = zk-alpha*grad-alpha*gamma*np.sign(z)
        #cost_list.append(g(z))
    return z

def dumb_prox_minimizer(zk,grad,alpha,gamma):
    return zk-alpha*grad
