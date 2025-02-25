# Clustering

The problem of clustering is that given a set of N objects $x_i$ each described by D values $x_{id}$ we wanto to find a *natural* partitioning in K cluster and, possibly, a number of *noise* object, the result is a *clustering scheme*, a function able to map each object into a category. 

The basic idea is that is necessary a **Centroid**, a point where the coordinates are computed as the average of the coordinates of all the points in the cluster.

It is possible to introduce a taxonomy of the clustering methods:

## K-means (Partitioning)

The K-means starts by deciding how many cluster we want, it can be a random number or choosen by the user, this number is not necessarerly the final one, it may changes as the algorithm goes to completion.

![](/Theory/Images/K-meansStart.png)

Once the temporary centers are choosen is necessary for each point to find the nearest center.

![](/Theory/Images/K-meansNearCenter.png)

Than for each center is necessary to find the centroid of its point, and then move towards the new assigned value.

The k-means starts working and the final result is the clustered dataset.

![](/Theory/Images/K-meansFinal.png)

On the slides is present the evolution of the K-means.

### How do we determine if the clustering was good?

It is possible to use some parameters to understand if the clustering was effective:

#### Distortion
Often called **Inertia**:

$ Distortion = \sum_{i=1}^{N}(x_i - c_{encode(x_i)})^2$

Here we define $x_i$ as part of the dataset and the conding function which is the encoding. 

The goal is to minimize the distortion, and to have the minimal distortion is necessary that each $x_i$ element must be encoded with the nearest center and that each center must be the **centroid** of the points it owns.

### Improving a sub-optimal solution

Sometimes is not possible to obtain the global optmal solution, but is possible to obtain the a good sub-optimal solution, and this solution is obtained when neather of the actions taken (center the point and move the centroid) makes any changes on the value of the distortion.

### Is the algorithm always going to end?

Given the fact that each point could be a centroid, and this is considered the 0 solution, in term of value. The algorithm ends with an approximate number of centers. Resulting in a non-global optimal solution.

### Establish the number of cluster
Given that the number of clusters are always insufficient to represent all the points, how is it possible to establish a good approximation number?

The solution is not fixed nor certain, it is possible to try with various values and use **quantitative evaluation**, the best value is the one that finds the optimal compromise between the minimization of the intra-cluster distance and the maxmization of the inter cluster distance.

Some extreme cases may happen anyway:
- Empty Clusters: if the centroid does not own any point, it is necessary to choose a new centroid
- Outliers: are points with a high distance to their centroid

### Common use of the K-means

Most of the times is used to explore data, in a one dimension space and it is a good way to discretize the values of the domain in non-uniform buckets

### Evaluation of a clustering scheme
The evaluation of a clustering technique is related on the results that it produces. Clustering is not a supervised method.

The main issue are related to the random apparent regularities, finding the optimal number of cluster.

The evaluation criteria are:
- Cohesion: it means that the proximity of the objects in the same cluster should be high
- Separation between two clusters

The sum of the proximity (another name for similarity) between the elements of the cluster and the geometric center.

- Centroid: a point in the space whose coordinate are the means of the dataset
- Medoid: an element of the dataset whose average dissimilarity with all the elements of the cluster is minimal

### Silhouette
A possible index to evaluate the clustering scheme is the Silhouette 

If the silhouette value is positive it means that the values is part of a cluster, instead if not it means that the object is more similar to other cluster's object.

### Choosin the best number of clusters
Given that some algorithm require to specify the number of cluster as an hyperparameter, for this reason it is necessary to iterate and reiterate the cluster training in order to obtain the best value of K.

### Supervised measure evaluation
The gold standard refers to a benchmark or a reference point used to evaluate the performance of algorithms or models. 
In clustering the gold standard migth involve validated relationship or grouping determined througth domain expertise.

## Hierarchical Clustering
It generates a **nested structure** of clusters which can be of two types:
- Agglomerative or bottom up
- Divisive or top down

### Agglomerative
Every object is a cluster and then we try to find the less separated pair and merge them into a single cluster, in the end the final result is a *dendogram*, a structure which represent the choosen cluster.

The parameter that can be used to discuss on the optimization of this method is the number of cluster that is necessary, based on that is possible to cut the choosen number of cluster.

![](/Theory/Images/ClusterCutting.png)

## Density based clustering

The cluster are caratherized by a high-density region compared to a low-density regions.

To compute the density is possible to use two methods:
- Grid-based
- Object-based

### Density Based Spatial Clustering of Application with Noise

The first goal is to start from the p and the q, the border and the core.

![](/Theory/Images/DBSCANStart.png)

After that is possible to define a radius and define the **neigthborhood** of a point.

![](/Theory/Images/DBSCANNeigthborhood.png)

Than it is possible to define a threshold $minPoints$ and define as core a point with at least $minPoints$ points in its neigthborhood.

A cluster is defined as a maximal set of point connected by density

To set the radius and the $minPoints$ is possible to use the *grid search* over several combination of hyperparameters.

## Kernel Based Estimation

Is technique developed in statistics and pattern mining and it describe the distribution of the data by a function, the overall density function is the sum of the **influence function** associated with each point.

This techniques is based upon the *kernel function*, it must be symmetric and monotonically decreasing.

