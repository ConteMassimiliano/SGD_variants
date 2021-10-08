import time
import numpy as np
import pandas as pd
import sklearn.datasets

np.random.seed(123)

# Preprocessing functions

def load_data(train_path, test_path):

	X_train, y_train, X_test, y_test = sklearn.datasets.load_svmlight_files(
	    (train_path, test_path))

	#for i in range(len(y_train)):
	#	if y_train[i] == -1:
	#		y_train[i] = 0

	#for i in range(len(y_test)):
	#	if y_test[i] == -1:
	#		y_test[i] = 0

	return np.array(pd.DataFrame.sparse.from_spmatrix(X_train)), y_train, np.array(pd.DataFrame.sparse.from_spmatrix(X_test)), y_test

def svrg(X, y, w0, epochs, iterations_inner_loop, stepsize, b, gradients):
    
    m, n = X.shape # Number of exampleas and features
    w = w0 # Initialize weights to zero
    w_list = []

    for epoch in range(1,epochs+1): # iterate over epochs
        
        past_w = w.copy()
        # w_history = np.zeros((iterations_inner_loop,n))
        # w_history[0] = w.copy()

        # v_0 = full gradient
        mu = gradients(X, y, w)

        for i in range(iterations_inner_loop):

            # random choose a sample 
            i_sample = np.random.randint(n, size = b)

            # compute accumulated direction
            # v = gradients(X[i_sample,:], y[i_sample], w_history[i-1]) - gradients(X[i_sample,:], y[i_sample], past_w) + mu
            v = gradients(X[i_sample,:], y[i_sample], w) - gradients(X[i_sample,:], y[i_sample], past_w) + mu

            # step
            w -= v * stepsize

            # w_history[i+1] = w.copy()

        # restart 
        # w = w_history[-1]
        w_list.append(w.copy())
  
    return w_list
    
# Metrics functions

def accuracy_metric(actual, predicted):
	'''
	Compute accuracy
	'''
	return np.sum(np.equal(actual, predicted))/len(actual)

# Plot loss funtions


def plot_loss(x_axes, loss_list, name_list, cpu=True):
	'''
	Plot multiple losses with legend
	'''
	import matplotlib.pyplot as plt
	import matplotlib.patches as mpatches

	fig, ax = plt.subplots(1, 1)
	fig.set_figheight(5)
	fig.set_figwidth(10)

	patch_list = []
	for j, name in enumerate(name_list):
		patch_list.append(mpatches.Patch(color=(j/len(x_axes), 0, 0), label=name))

	if cpu:
		ax.set_title("Cost function vs cpu time", size=20)
		ax.set_xlabel("Time (in seconds)", size=10)
	else:
		ax.set_title("Cost function vs epochs", size=20)
		ax.set_xlabel("Epochs", size=10)
	ax.set_ylabel("Cost function", size=10)

	for j in range(len(x_axes)):
		ax.plot(x_axes[j], loss_list[j], c=(j/len(x_axes), 0, 0))
	_ = ax.legend(handles=patch_list)


def plot_loss_log_scale(loss_list, name_list, title,x_lim, y_lim = [1e-4, 1e1]):
    '''
    Plot multiple losses with legend on the log scale
    y_lim = None means no limits
    '''
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    x_axes = [list(np.arange(len(l))) for l in loss_list]

    fig, ax = plt.subplots(1, 1)
    ax.set_yscale('log', base = 10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    ax.set_xlim(x_lim)
    if y_lim != None:
        ax.set_ylim(y_lim)
        
    patch_list=[]
    
    colors = ["r", "b", "g", "k", "c", "v"]

    for j, name in enumerate(name_list):
        patch_list.append(mpatches.Patch(
            color=colors[j], label=name))
    
    ax.set_title(title, size=20)
    ax.set_xlabel("# of epochs", size=10)
    ax.set_ylabel("Loss(l-l*)", size=10)

    for j in range(len(x_axes)):
        ax.plot(x_axes[j], loss_list[j], c=colors[j])
        
    for j in range(len(x_axes)):
        for i in range(len(loss_list[j])):
            #print(x_axes[j][i])
            if loss_list[j][i] < y_lim[0]:
                ax.hlines( y = y_lim[0], xmin = x_axes[j][i], xmax = x_axes[j][-1], color=colors[j], linestyle='-')
    _=ax.legend(handles=patch_list)


def plot_loss(loss_list, name_list, y_lim = [1e-4, 1e1]):
    '''
    Plot multiple losses with legend on the log scale
    y_lim = None means no limits
    '''
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    x_axes = [list(np.arange(len(l))) for l in loss_list]

    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    if y_lim != None:
        ax.set_ylim(y_lim)
        
    patch_list=[]
    
    colors = ["r", "b", "g", "k", "c", "v"]

    for j, name in enumerate(name_list):
        patch_list.append(mpatches.Patch(
            color=colors[j], label=name))
    
    ax.set_title("Cost function vs epochs", size=20)
    ax.set_xlabel("Epochs", size=10)
    ax.set_ylabel("Cost function", size=10)

    for j in range(len(x_axes)):
        ax.plot(x_axes[j], loss_list[j], c=colors[j])
    _=ax.legend(handles=patch_list)

def loss_lr_smooth_reg(X,y,w,gamma = 0.1):
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    
    y_hat = sigmoid(np.dot(X,w))
    loss = -np.mean(y*(np.log(y_hat)+1e-20) + (1-y)*np.log(1-y_hat+1e-20))
    loss += gamma*np.sum(w**2/(1+w**2))
    return loss

def loss_lr_smooth_reg_minus_one_notation(X,y,w,gamma = 0.1):    
    loss = np.mean(y*(np.log(1 + np.exp(-np.dot(X,w)))))
    loss += gamma*np.sum(w**2/(1+w**2))
    return loss

def loss_robust_smooth_reg(X,y,w,gamma = 0.1): 
    error = y - np.dot(X,w)
    loss = np.mean(np.log(error**2 / 2 + 1))
    return loss