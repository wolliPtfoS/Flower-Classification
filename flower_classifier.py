import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import math 

iris = datasets.load_iris()

X = iris.data
y = iris.target

# x1 = sepal length / sepal width
# x2 = petal length / petal width
for i in range(len(X)):
    X[i][0] /= X[i][1]
    X[i][2] /= X[i][3]

# plot our 
data = X[:, [0, 2]]
plt.scatter(data[:, 0], data[:, 1], c = y, edgecolors = 'purple')
plt.title('X2 vs X1, distinguised by color')
plt.show()

def k_init(data, k):
    """ k-means++: initialization algorithm

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    Returns
    -------
    init_centers: array (k, d)
        The initialize centers for kmeans++
    """
    cens = []
    cens.append(data[np.random.randint(data.shape[0]), :])
  
    ## find our other centers
    for num in range(k - 1):
         
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            ds = math.dist(point, cens[0])
             
            ## find newest min distance, loop and update
            for j in range(len(cens)):
                temp = math.dist(point, cens[j])
                ds = min(ds, temp)
            dist.append(ds)
             
        ## next centroid is one with largest dist
        dist = np.array(dist)
        newer = data[np.argmax(dist), :]
        cens.append(newer)
        dist = []
    return cens
                

def assign_data2clusters(X, C):
    """ Assignments of data to the clusters
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    data_map: array, shape(n, k)
        The binary matrix A which shows the assignments of data points (X) to
        the input centers (C).
    """
    # useful for later
    lenc = range(len(C))
    lenx = range(len(X))
    clust = np.zeros(len(data))

    # zero out our map
    mapping = [[0 for c in lenc] for x in lenx]
    # copy our data map
    dst = mapping
    
    for i in lenx:
        
        # need our coords distance from other centers
        for j in lenc:
            
            dst[i][j] = math.dist(X[i], C[j])
            
        for k in lenc:
            
            dst_lst = np.array(dst[i])
            
        # argmin
        clust[i] = np.argmin(dst_lst)
        
    clust = clust.astype(int)
    
    # map our data
    for i in lenx:
        for j in lenc:
            if (clust[i] == j):
                
                mapping[i][j] = False
                
    return mapping
                
            


def compute_objective(X, C):
    """ Compute the clustering objective for X and C
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    accuracy: float
        The objective for the given assigments
    """
    lenc = range(len(C))
    lenx = range(len(X))
    
    total = 0
    
    obj = [[0 for c in lenc] for x in lenx]

    # loop through, find distances, and then square our sums after they're compiled for each iteration
    for i in lenx:
        
       for j in lenc:
           
           obj[i][j] = math.dist(X[i], C[j])
           
       for k in lenc:
           
           dst = np.array(obj[i])
           total += (min(dst)) ** 2
           
    return total


def k_means_pp(X, k, max_iter):
    """ k-means++ clustering algorithm

    step 1: call k_init() to initialize the centers
    step 2: iteratively refine the assignments

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    max_iter: int
        Maximum number of iteration

    Returns
    -------
    final_centers: array, shape (k, d)
        The final cluster centers
    objective_values: array, shape (max_iter)
        The objective value at each iteration
    """
    # intializer
    lenx = range(len(X))
    start_centers = k_init(X, k)
    obj = []

    # Assign our data
    mapping = assign_data2clusters(X, start_centers)
    clust = np.zeros(len(data))
    final_centers = start_centers
    
    # Calculate initial objective
    obj.append(compute_objective(X, final_centers))
                     
    # Loop through, applying our new points to each iteration
    for i in range(max_iter):
        
        mapping = assign_data2clusters(X, final_centers)

        for a in lenx:
            
            for b in range(k):
                
                if (mapping[a][b] == False):
                    
                    clust[a] = b
                    
        for a in range(k):
            
            vals = [X[b] for b in lenx if clust[b] == a]
            final_centers[a] = np.mean(vals, axis=0)
            
        # Objective
        obj.append(compute_objective(X, final_centers))
        
    # plot results
    plt.plot(obj)
    plt.title('Objective (y-axis) vs Iteration Number (x-axis)')
    plt.show()
    plt.clf()

    return final_centers

# Plot accuracies
cens = []
obj = []

for k in range(1, 6):
    cens.append(k_means_pp(data, k, 50))
    
for k in range(1, 6):
    obj.append(compute_objective(data, cens[k-1]))

# Clean up our display
plt.plot(range(1, len(obj) + 1), obj)
plt.xticks(range(1, len(obj) + 1))
plt.title('Accuracy Plot: Objective (y-axis) vs Cluster Number (x-axis)')
plt.show()


plt.clf()

# Final plot, utilizing k = 3
center_map = assign_data2clusters(data, cens[2])
clust = np.zeros(len(data))
lend = range(len(data))


for a in lend:
    for b in range(len(cens[2])):
        if center_map[a][b] == False:
            clust[a] = b

points = [data[a] for a in lend if clust[a] == 0]
points = np.array(points)
plt.scatter(points[:, 0], points[:, 1], c='#32CD32', edgecolors= 'purple')

points = [data[b] for b in lend if clust[b] == 1]
points = np.array(points)
plt.scatter(points[:, 0], points[:, 1], c='red', edgecolors= 'purple')

points = [data[c] for c in lend if clust[c] == 2]
points = np.array(points)
plt.scatter(points[:, 0], points[:, 1], c='cyan', edgecolors= 'purple')

thing = np.array(cens[2])
plt.scatter(thing[:, 0], thing[:, 1], s = 150, c='black', marker='*')
plt.title('Final Plot')
plt.show()
