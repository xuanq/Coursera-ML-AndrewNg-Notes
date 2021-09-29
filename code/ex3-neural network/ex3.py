import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from multiprocessing import Pool

import plotly.express as px
import plotly.graph_objects as go

# # 前100个 排列成10*10 (200*200)
# X_plot = X[0:5000:50]
# digits = [X_plot[i].reshape(20,20).T for i in range(100)]

# rows_digits = []
# for row in range(10):
#     rows_digits.append(np.hstack(digits[row*10:row*10+10]))

# digits_img = np.vstack(rows_digits)

# fig = px.imshow(digits_img,color_continuous_scale='gray')
# fig.show()

# g(theta,x)
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def h(theta,x):
    return sigmoid(np.dot(x,theta))

# J(Theta)
def cost(theta, x, y):
    cost = -y*np.log(h(theta, x)) - (1-y)*np.log(1-h(theta, x))
    return np.mean(cost)

def gradients(theta,x,y):
    return np.mean(x.T*(h(theta,x)-y),axis=1)

def regularized_cost(theta, x, y, l=1):
    n=len(y)
    theta_1_to_k = theta[1:]
    regular_term = l*(theta_1_to_k@theta_1_to_k)/(2*n)
    return cost(theta,x,y)+regular_term

def regularized_gradients(theta,x,y,l=1):
    n = len(y)
    regularized_terms = theta*l/n
    regularized_terms[0]=0
    return gradients(theta, x, y)+regularized_terms

def main():
    data = loadmat("ex3data1.mat")

    X = data["X"]
    y = data["y"]
    
    X = np.hstack([np.ones(len(y)).reshape(-1,1),X])
    pool = Pool(processes=5)
    K=10
    results = []
    thetas = []
    for i in range(1,K+1):
        y_mask = ((y == i)*1).flatten()
        theta0 = np.zeros(X.shape[1])
        # res = minimize(fun=regularized_cost, x0=theta0, args=(X, y_mask), method='Newton-CG', jac=regularized_gradients)
        result = pool.apply_async(minimize,kwds={'fun':regularized_cost,"x0":theta0,"args":(X, y_mask),"method":'Newton-CG',"jac":regularized_gradients})
        results.append(result)
    pool.close()
    pool.join()

    for res in results:
        thetas.append(res.get().x)
    thetas = np.vstack(thetas)
    
    np.savetxt("thetas.csv", thetas, delimiter=",")

if __name__ == "__main__":
    main()