import numpy as np
'''
Loss and gradient for (-1,1) lr (no paper version)
'''
def gradient_lr_np(X,y,w, gamma = 0.1):
    m = X.shape[0]
    y_hat = np.dot(X,w)
    coeff = -y/(1+np.exp(y*y_hat))
    dw = (1/m)*np.dot(X.T, coeff)
    return dw

def loss_lr_np(X,y,w, gamma = 0.1):    
    loss = np.mean((np.log(1 + np.exp(-y*np.dot(X,w)))))
    return loss

'''
Loss and gradient for (-1,+1) logistic regression (no paper version) and nc smooth regolarizer
'''
def loss_lr_np_nc(X,y,w,gamma = 0.1):    
    loss = loss_lr_np(X,y,w)
    loss += gamma*np.sum(w**2/(1+w**2))
    return loss

def gradient_lr_np_nc(X,y,w,gamma = 0.1):
    dw = gradient_lr_np(X,y,w)
    dw += 2*gamma*w/((1+w**2)**2)
    return dw

'''
Loss and gradient for (-1,+1) logistic regression (no paper version) and convex smooth regolarizer
'''
def loss_lr_np_convex(X,y,w,gamma = 0.1):    
    loss = loss_lr_np(X,y,w)
    loss += gamma*np.linalg.norm(w)**2
    return loss

def gradient_lr_np_convex(X,y,w,gamma = 0.1):
    dw = gradient_lr_np(X,y,w)
    dw += 2*gamma*w
    return dw

'''
Loss and gradient for (-1,+1) logistic regression (no paper version) and l1 regolarizer
'''
def loss_lr_np_l1(X,y,w,gamma = 0.1):    
    loss = loss_lr_np(X,y,w)
    loss += gamma*np.linalg.norm(w, 1)
    return loss

def gradient_lr_np_l1(X,y,w,gamma = 0.1):
    dw = gradient_lr_np(X,y,w)
    dw += gamma*np.sign(w)
    return dw

'''
Loss and gradient for (-1,1) lr (paper version)
'''
def gradient_lr_pv(X,y,w, gamma = 0.1):
    m = X.shape[0]
    y_hat = np.dot(X,w)
    coeff = -y/(1+np.exp(y_hat))
    dw = (1/m)*np.dot(X.T, coeff)
    return dw

def loss_lr_pv(X,y,w, gamma = 0.1):    
    loss = np.mean(y*(np.log(1 + np.exp(-np.dot(X,w)))))
    return loss

'''
Loss and gradient for (-1,+1) logistic regression (paper version) and nc smooth regolarizer
'''
def loss_lr_pv_nc(X,y,w,gamma = 0.1):    
    loss = loss_lr_pv(X,y,w)
    loss += gamma*np.sum(w**2/(1+w**2))
    return loss

def gradient_lr_pv_nc(X,y,w,gamma = 0.1):
    dw = gradient_lr_pv(X,y,w)
    dw += 2*gamma*w/(1+w**2)**2
    return dw

'''
Loss and gradient for (-1,+1) logistic regression (no paper version) and convex smooth regolarizer
'''
def loss_lr_pv_convex(X,y,w,gamma = 0.1):    
    loss = loss_lr_pv(X,y,w)
    loss += gamma*np.linalg.norm(w)**2
    return loss

def gradient_lr_pv_convex(X,y,w,gamma = 0.1):
    dw = gradient_lr_pv(X,y,w)
    dw += 2*gamma*w
    return dw


'''
Loss and gradient for robust regression
'''

def loss_rr(X,y,w, gamma = 0.1):
    return np.mean(np.log(0.5*(y-np.dot(X,w))**2 + 1 ))

def gradient_rr(X,y,w, gamma = 0.1):
    m = X.shape[0]
    diff = y-np.dot(X,w)
    coeff = -diff/(0.5*diff**2+1)
    dw = (1/m)*np.dot(X.T, coeff)
    return dw

'''
Loss and gradient for robust regression and convex regolarizer
'''

def loss_rr_convex(X,y,w,gamma = 0.1):
    loss = loss_rr(X,y,w)
    loss += gamma*np.linalg.norm(w)**2
    return loss

def gradient_rr_convex(X,y,w,gamma = 0.1):
    dw = gradient_rr(X,y,w)
    dw += 2*gamma*w
    return dw

'''
Loss and gradient for robust regression and l1 regolarizer
'''

def loss_rr_l1(X,y,w,gamma= 0.1):
    loss = loss_rr(X,y,w)
    loss += gamma*np.linalg.norm(w,1)
    return loss

def gradient_rr_l1(X,y,w,gamma = 0.1):
    dw = gradient_rr(X,y,w)
    dw += (gamma*np.sign(w))
    return dw