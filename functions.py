def create_blocks(block_length, maximum):
    '''
    Return a list of lists of ordered natural numbers 
    for example:
    block_length = 2
    maximum = 5
    returns the list  [ [0,1] , [2,3] , [4] ]
    '''
    blocks = []
    for len in range(0, maximum, block_length):
        if (len+block_length) > maximum:
            blocks.append(list(range(len,maximum)))
        else:
            blocks.append(list(range(len,len+block_length)))
    return blocks

def distance_matrix(X_labeled,X_unlabeled, order = 2):
    '''
    given X_labeled and X_unlabeled matrices of l and mu n-dimensional
    vectors respectively (stored in rows), returns a matrix W of the euclidean distances
    between the vectors in X_unlabeled and the ones in X_labeled and X_unlabeled,
    such that:
    W[i,j] = ||X_unlabled[i,:]-X_labeled[j,:]|| 
    W[i,j+l] = ||X_labeled[i,:]-X_labeled[j,:]||
    '''
    import numpy as np

    l = X_labeled.shape[0]
    u = X_unlabeled.shape[0]
    W = np.zeros([u,u+l])
    for i in range(u):
        for j in range(l): 
            W[i,j] = np.linalg.norm(X_unlabeled[i]-X_labeled[j], ord = order)
        for j in range(u):
            W[i,j+l] = np.linalg.norm(X_unlabeled[i]-X_unlabeled[j], ord = order)
    return W

def normalize_weight_matrix(W,y_labeled,data_bias = 1):
    '''
    Given a weight matrix W returns a matrix with normalized weights on the labeled data 
    so to have a 50/50 representation for both classes of clusters.
    It also rebalances the weights so to have a 50/50 representation of labeled and unlabeled data.
    If you want to have a N:1 representation of labeled and unlabeled data just set data_bias = N
    '''
    import numpy as np

    bias_1 = int(sum(y_labeled == 1))/len(y_labeled)
    bias_2 = 1-bias_1
    coeff = (1./2.)*data_bias*float(W.shape[1]-len(y_labeled))/len(y_labeled)
    b1 = coeff/bias_1
    b2 = coeff/bias_2
    W_norm = W.copy()
    for j in range(len(y_labeled)):
        W_norm[:,j] = W_norm[:,j]*((y_labeled[j]==1)*b1 + (b2)*(y_labeled[j]==-1))
    return W_norm

def cost_function(y_unlabeled, y_labeled, W):
    '''
    Given the weight matrix W,the known labels y_known and the predicted labels y_unknown
    returns the cost function of the semisupervised problem
    '''
    import numpy as np

    unlabeled_matrix_1 = y_unlabeled @ np.ones(y_labeled.shape[0]).reshape(1,-1)
    unlabeled_matrix_2 = y_unlabeled @ np.ones(y_unlabeled.shape[0]).reshape(1,-1)
    labeled_matrix = np.ones(y_unlabeled.shape[0]).reshape(-1,1) @ y_labeled.reshape(1,-1)

    y_ul = (unlabeled_matrix_1 - labeled_matrix)**2
    y_uu = 1/2 * (unlabeled_matrix_2 - unlabeled_matrix_2.T)**2

    cost = (W * np.concatenate((y_ul,y_uu),axis=1)).sum()
    
    return float(cost)

def plot_data(data, label):
    '''
    '''
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig,ax= plt.subplots(1,2)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    red_patch = mpatches.Patch(color='red', label=label['class'][0])
    blue_patch = mpatches.Patch(color='blue', label=label['class'][1])
    black_patch = mpatches.Patch(color='black', label='Unlabeled')

    ax[0].scatter(data['x1 labeled'], data['x2 labeled'], c = ['red' if l == -1 else 'blue' for l in data['y labeled']])
    ax[0].scatter(data['x1 unlabeled'], data['x2 unlabeled'], c = ['red' if l == -1 else 'blue' for l in data['y unlabeled']])
    ax[0].set_title("All data",size = 24)
    ax[0].set_xlabel(label['x1'],size = 15) 
    ax[0].set_ylabel(label['x2'],size = 15) 
    ax[0].legend(handles=[red_patch, blue_patch])

    ax[1].scatter(data['x1 labeled'], data['x2 labeled'], c = ['red' if l == -1 else 'blue' for l in data['y labeled']])
    ax[1].scatter(data['x1 unlabeled'], data['x2 unlabeled'], c = "black")
    ax[1].set_title("Labeled vs Unlabeled data",size = 24)
    ax[1].set_xlabel(label['x1'],size = 15) 
    ax[1].set_ylabel(label['x2'],size = 15) 
    _=ax[1].legend(handles=[red_patch, blue_patch,black_patch])

def plot_loss(time_list, loss_list):
    '''
    '''
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig,ax= plt.subplots(1,1)
    fig.set_figheight(10)
    fig.set_figwidth(20)

    red_patch = mpatches.Patch(color='red', label='Gradient Descent')
    blue_patch = mpatches.Patch(color='blue', label='BCGD randomized')
    black_patch = mpatches.Patch(color='black', label='BCGD cyclic')

    ax.set_title("Cost function vs cpu time",size = 24)
    ax.set_xlabel("Time (in seconds)",size = 12)
    ax.set_ylabel("Cost function",size = 12)
    ax.plot(time_list[0],loss_list[0], c='red')
    ax.plot(time_list[1],loss_list[1], c='blue')
    ax.plot(time_list[2],loss_list[2], c='black')

    _= ax.legend(handles=[red_patch, blue_patch, black_patch])

def plot_loss_but_better(time_list, loss_list,name_list):
    '''
    '''
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    fig,ax= plt.subplots(1,1)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    patch_list = []
    for j,name in enumerate(name_list):
        patch_list.append(mpatches.Patch(color=(j/len(time_list),0,0), label=name)) 
    ax.set_title("Cost function vs cpu time",size = 24)
    ax.set_xlabel("Time (in seconds)",size = 12)
    ax.set_ylabel("Cost function",size = 12)
    for j in range(len(time_list)):
        ax.plot(time_list[j],loss_list[j], c=(j/len(time_list),0,0))
    _= ax.legend(handles=patch_list)

def plot_performace(y_best, label, data, title):
    '''
    '''
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig,ax= plt.subplots(1,2)
    red_patch = mpatches.Patch(color='red', label=label['class'][0])
    blue_patch = mpatches.Patch(color='blue', label=label['class'][1])
    fig.set_figheight(10)
    fig.set_figwidth(20)
    
    ax[0].scatter(data['x1 unlabeled'], data['x2 unlabeled'],c=["r" if val <0 else "b" for val in y_best])
    ax[1].scatter(data['x1 unlabeled'], data['x2 unlabeled'],c=["r" if val <0 else "b" for val in data['y unlabeled']])
    ax[0].set_xlabel(label['x1'],size =15) 
    ax[1].set_ylabel(label['x2'],size =15)
    ax[1].set_xlabel(label['x1'],size =15)
    ax[0].set_ylabel(label['x2'],size =15)
    ax[0].legend(handles=[red_patch, blue_patch])
    ax[1].legend(handles=[red_patch, blue_patch])
    ax[0].set_title("Predicted Data",size = 24)
    _=ax[1].set_title("True Data",size = 24)