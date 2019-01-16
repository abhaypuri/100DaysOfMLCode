import numpy as np

def initialize(X,Y):
    W = np.random.randn(Y.shape[0],X.shape[0])
    b = np.zeros((Y.shape[0],1))
    return W,b

def softmax_activation(Z):
    return np.exp(Z) / np.sum(np.exp(Z),axis=0)

def forward_prop(X, W, b, activationfn=softmax_activation):
    Z = np.dot(W,X)+b
    A = activationfn(Z)
    return (A,Z)

def compute_cost(A, Y, W, lambd):
    m = Y.shape[1]
    J = (-1/m) * np.sum(np.multiply(np.log(A),Y)) + ((lambd/(2*m)) * np.sum(np.square(W)))
    return J

def back_prop(X, Y, forward_cache):
    m = X.shape[1]
    A,Z = forward_cache
    dW = (-1/m) * np.dot((Y-A),X.T)
    db = (-1/m) * np.sum((Y-A),axis=1)
    db = db.reshape(db.shape[0],1)
    return (dW,db)

def update_params(W, b, backward_cache, alpha, lambd, m):
    dW,db = backward_cache
    W = W - alpha * (dW+(lambd/m)*W)
    b = b - alpha * db
    return (W,b)

def train(X, Y, epochs = 1000, alpha = 0.01, lambd = 0.1, displayRate = 100):
    W,b = initialize(X,Y)
    costs = []
    iters = []
    
    for i in range(epochs):
        #print('Iteration: ',i+1)
        forward_cache = forward_prop(X, W, b, activationfn = softmax_activation)
        A,Z = forward_cache
        cost = compute_cost(A,Y,W,lambd)
        costs.append(cost)
        iters.append(i+1)
        if i % displayRate==0:
            print('Iteration ',i,' Cost: ',cost)
        backward_cache = back_prop(X, Y, forward_cache)
        dW,db = backward_cache
        '''
        print('W.shape: ',W.shape)
        print('b.shape: ',b.shape)
        print('A.shape: ',A.shape)
        print('Z.shape: ',Z.shape)
        print('dW.shape: ',dW.shape)
        print('db.shape: ',db.shape)
        '''
        W,b = update_params(W, b, backward_cache, alpha, lambd, X.shape[1])
    return (W,b,costs,iters)

def predict(X, W, b, activationfn = softmax_activation):
    A, Z = forward_prop(X, W, b, activationfn)
    maxi = np.max(A,axis=0)
    for i in range(len(maxi)):
        for j in range(A.shape[0]):
            if(A[j,i] == maxi[i]):
                A[j,i] = 1
            else:
                A[j,i] = 0
    return A

def evaluate_accuracy(A, Y):
    count = 0
    for i in range(A.shape[1]):
        if np.array_equal(A[:,i], Y[:,i]):
            count+=1
    return count/Y.shape[1]
