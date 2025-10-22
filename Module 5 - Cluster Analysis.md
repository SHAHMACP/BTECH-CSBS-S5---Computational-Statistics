# **Cluster Analysis**

**Cluster Analysis** is a statistical technique used to group a set of similar objects into clusters such that:

* Objects within the same cluster are **highly similar** (high intra-cluster similarity), and
* Objects in different clusters are **dissimilar** (low inter-cluster similarity).

It is one of the primary methods in **unsupervised learning**, where the goal is to discover patterns or structures within unlabeled data.

<img width="312" height="162" alt="image" src="https://github.com/user-attachments/assets/83a17911-b932-480c-8e0d-f87b214fd380" />

Clustering is a fundamental step in data mining, pattern recognition, and machine learning.
It helps identify hidden relationships in data by dividing it into meaningful subgroups (clusters) based on similarity.

* Each cluster represents a group of data points sharing common characteristics.
* Clustering can be used for exploratory data analysis, summarization, anomaly detection, and more.

Example: Grouping customers by purchasing behavior, documents by topic, or genes by expression patterns.

## Dunn Index

Dunn index is the ratio of the minimum of inter-cluster distances and maximum of intracluster distances. 
The more the value of the Dunn index, the better the clusters will be.
![image](https://github.com/user-attachments/assets/c25f5868-6cd0-446a-beec-5ba8242e142d)

| Type                         | Description                                              | Goal               |
| ---------------------------- | -------------------------------------------------------- | ------------------ |
| **Intra-Cluster Similarity** | Similarity among objects **within** the same cluster.    | Should be **high** |
| **Inter-Cluster Similarity** | Similarity among objects **between** different clusters. | Should be **low**  |

Thus, a good clustering method aims to:

* Maximize intra-cluster similarity
* Minimize inter-cluster similarity


---


## **Advantages of Cluster Analysis**

* **No need for labeled data:** Works without predefined class labels.
* **Reveals hidden structures:** Finds natural groupings in data.
* **Flexible:** Can be applied to various data types and domains.
* **Preprocessing support:** Useful for summarizing data before classification or visualization.

---

## **Applications of Cluster Analysis**

* **Market Segmentation**: Businesses use cluster analysis to group customers based on behavior, preferences, demographics, or purchasing patterns. This helps in creating targeted marketing strategies and personalized product offerings.
* **Image Segmentation**: In computer vision, clustering is used to identify regions within images for tasks like object recognition, medical imaging (e.g., tumor detection), or face detection.
* **Fraud Detection**: Banks and financial institutions use cluster analysis to identify unusual transactions that may indicate fraud. By detecting outliers or abnormal patterns, they can flag potential fraud cases.
* **Social Network Analysis**: Clustering is used to analyze relationships and communities within social networks, grouping individuals or entities based on similar interests, connections, or communication patterns.
* **Document Classification**: In text mining, cluster analysis helps group similar documents or articles. This is useful in organizing large sets of unstructured data like research papers, news articles, or customer feedback.
* **Biological Data Analysis**: In genomics and bioinformatics, clustering helps identify groups of genes or proteins with similar functions, aiding in the study of diseases, genetic traits, or drug responses.
* **City Planning**: Governments and urban planners use clustering to identify regions with similar population densities, traffic patterns, or economic activities, helping in efficient resource allocation and urban development.
* **Recommender Systems**: E-commerce and streaming platforms use clustering to group users based on their behavior (like purchases or viewing history), providing personalized recommendations.

---

## **Types of Clustering Methods**
<img width="1447" height="1071" alt="image" src="https://github.com/user-attachments/assets/b7180539-e39f-4c06-a067-4065ac944ee9" />


| Type                         | Description                                                                             | Examples                                          |
| ---------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------- |
| **Hierarchical Clustering**  | Builds a hierarchy of clusters by successively merging or splitting clusters. It can be visualized as a tree. No need to specify the number of clusters in advance         | Agglomerative, Divisive |
| **Partitioning Clustering**  | Divides data into a fixed number of non-overlapping clusters. Construct various partitions and then evaluate them by some criterion. The number of clusters should be defined beforehand. The algorithm typically tries to minimize the distance between data points and their corresponding cluster centers.                     | K-Means, K-Medoids                                |
| **Density-Based Clustering** | Forms clusters based on regions of high density separated by low-density areas.         | DBSCAN, OPTICS                                    |
| **Overlapping Clustering**   | Data points can belong to more than one cluster. It allows more flexible grouping where points can have partial membership to different clusters.         | Gaussian Mixture Models (GMM)  , Fuzzy C means                   |   

---
## **Common Distance Metrics**
- **Euclidean Distance** :  $d(x, y) = \sqrt{\sum_i (x_i - y_i)^2} =  \sqrt{ (x_1 - y_1)^2+ (x_2 - y_2)^2+...(x_n - y_n)^2}$
- **Manhattan Distance** : $d(x, y) = \sum_i| x_i - y_i| = | x_1 - y_1|+| x_2 - y_2|+...+| x_n - y_n|$
- **Minkowski Distance** : $d(x, y) = (\sum_i| x_i - y_i| ^p)^{1/p}$
- **Chebyshev Maximum Distance** : $d(x, y) = max(| x_i - y_i|) = max(| x_1 - y_1|,| x_2 - y_2|,...,| x_n - y_n|)$ 

---

## **Overlapping vs Non-Overlapping Clustering**

### **Overlapping Clustering**

* Allows data points to belong to multiple clusters simultaneously.
* Unlike traditional clustering methods, where each data point is assigned to a single cluster, overlapping clustering recognizes that many real-world data points may naturally fit into more than one category.
* This approach is particularly useful in scenarios where categories are not mutually exclusive

  #### Example:
  In a movie recommendation system, consider four users with varying preferences for genres: User 1 enjoys Action (70%) and Comedy (30%), User 2 prefers Action (60%) and Drama (40%), User 3 likes Comedy (50%) and Drama (50%), while User 4 exclusively enjoys Action (80%).
Using overlapping clustering, such as Fuzzy C-Means, we can assign users to multiple genres. For instance, User 1 belongs primarily to the Action cluster but also partially to the Comedy cluster. This approach allows the system to recommend movies that cater to each user's diverse interests across multiple genres, rather than restricting them to a single category 

#### Features
1. Partial Membership: Each data point can belong to multiple clusters with different degrees of membership.
   
2. Flexibility: It allows for a more realistic representation of complex datasets, accommodating ambiguity and shared characteristics.

3. Use Cases: Ideal for applications such as social network analysis, recommendation systems, and text categorization.

#### Common Algorithms
##### **(1) Fuzzy C-Means** (FCM):
* Fuzzy C-Means is one of the most widely used overlapping clustering methods.
* Instead of assigning each data point to one cluster, it assigns a degree of membership to each data point for every cluster.
* Each data point has a membership value between 0 and 1, representing its degree of belonging to each cluster.

##### **(2) Probabilistic Clustering** (Gaussian Mixture Models - GMM):

* In GMM, each cluster is modeled as a Gaussian distribution, and the probability that a data point belongs to a particular cluster is calculated.
* Data points can belong to multiple clusters with different probabilities
  
### **Non-Overlapping (Exclusive / Partitioning) Clustering** 

Each object belongs to exactly one cluster. Each data point belongs to one cluster only and the objective is to optimize the placement of points in these clusters based on some criteria, typically minimizing the distance between the points and their respective cluster centroids.

#### Features

* **Fixed Number of Clusters**: The user must specify the number of clusters (k) in advance.
* **Non-overlapping Clusters**: Each data point is assigned to one cluster only.
* **Iterative Process**: The algorithms often involve iterations to refine the clusters based on distances.
* **Simplicity**: Easy to understand and implement.
* **Speed**: Generally faster than hierarchical methods, especially for large datasets.
* **Scalability**: Efficient for large datasets when using algorithms like K-Means.

  #### Common Algorithms

##### **(1) K-Means Clustering**:

* One of the most popular partitioning clustering methods.
* The algorithm randomly selects k initial centroids and assigns each data point to the nearest centroid.
* The centroids are then recalculated as the mean of the points in each cluster, and the process repeats until convergence.
* Suitable for large datasets but assumes clusters are spherical and equally sized.
* Example: In customer segmentation, K-Means might identify distinct groups of customers based on purchasing behavior, such as "frequent buyers," "occasional buyers," and "one-time buyers

##### **(2) K-Medoids Clustering**:

* Similar to K-Means, but instead of centroids, it uses actual data points (medoids) as the center of clusters.
* More robust to noise and outliers than K-Means.
* The algorithm iterates to find the medoids that minimize the total distance to all points in the cluster.

##### **(3) CLARA (Clustering LARge Applications)**:

* An extension of K-Medoids designed for larger datasets.
* It randomly samples a subset of data points to form medoids and applies K-Medoids to these samples.
* The clusters are then evaluated and refined based on the entire dataset.
---

# K-Means Clustering

**K-Means Clustering** is a **partitioning-based unsupervised learning algorithm** used to group data points into *k* clusters based on feature similarity.  
Each cluster is represented by its **centroid**, which is the mean position of all the points in that cluster.


## Definition

K-Means clustering aims to divide *n* observations into *k* clusters such that each observation belongs to the cluster with the **nearest mean (centroid)**.

> **Objective:** Minimize the total intra-cluster distance (compactness of clusters).

---

## Algorithm Steps

1. **Initialize** the number of clusters *k*.  
2. **Select k initial centroids** randomly from the dataset.  
3. Calculate the **distance** between each data point and each cluster centroid.
4. **Assign each point** to the nearest centroid (based on Euclidean distance) for which the distance is minimum.  
5. **Recalculate each centroid** as the mean of all points assigned to each cluster.  
6. **Repeat** steps 3–5 until:
   - Centroids no longer move, or  
   - No reassignment of data points happened.

---

## Objective Function

The K-Means algorithm minimizes the **Within-Cluster Sum of Squares (WCSS)**:

$$
J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
$$

where:
- $C_i$ → cluster i  
- $\mu_i$ → centroid of cluster i  
- $||x - \mu_i||^2$ → squared distance between data point and centroid  

---

## Distance Measure

The most common distance metric used in K-Means is the **Euclidean distance**:

$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

---

## Properties of K-Means

| Property | Description |
|-----------|--------------|
| **Type** | Partitioning, Unsupervised Learning |
| **Shape of clusters** | Spherical / Convex |
| **Cluster representation** | By centroids (mean of points) |
| **Objective** | Minimize within-cluster variance |
| **Output** | Cluster labels and centroids |
| **Scalability** | Efficient for medium to large datasets |
| **Distance metric** | Usually Euclidean distance |

---

## Advantages

- Simple and easy to implement  
- Scales well with large datasets  
- Works well with well-separated spherical clusters  
- Produces easily interpretable results  

---

## Limitations

- Must predefine the number of clusters k 
- Sensitive to initial centroid positions  
- Struggles with non-spherical or unequal-sized clusters  
- Sensitive to outliers and noise  

---

## Improving K-Means Results

1. **Optimal k selection:**  Use the **Elbow Method** or **Silhouette Score** to find the right number of clusters.  

2. **Multiple initializations:**  Run K-Means multiple times with different random centroids and choose the lowest error.  

3. **Data normalization:**  Standardize features before clustering to avoid bias due to scale differences.  

4. **Outlier handling:**  Detect and remove outliers that can distort cluster centers.  

5. **Dimensionality reduction:**  Use **PCA** to improve performance on high-dimensional data.

---

## Methods to Find Optimal Number of Clusters (k)

### **a. Elbow Method**
Plot *k* vs. WCSS and choose the “elbow point” where the decrease slows down.
![image](https://github.com/user-attachments/assets/799c1b7a-48ba-4563-a5a2-d959976871b3)

### **b. Silhouette Score**
Measures how well each point fits within its cluster.  
Range: `-1 to +1`  
Higher values indicate better clustering.

---

## Interpreting K-Means Clusters

- **Centroids:** Represent the “average” feature values in each cluster.  
- **Feature Comparison:** Compare centroid values to understand what defines each cluster.  
- **Visualization:** Scatter plots, bar plots, and 3D visualizations help interpret results.

---

## Example

 ### Problem 1

 Use K-Means clustering algorithm to divide the following data into two clusters
  |  X |  Y |
| -: | -: |
|  1 |  1 |
|  2 |  1 |
|  2 |  3 |
|  3 |  2 |
|  4 |  3 |
|5   |  5 |

> **Given points:** (1,1), (2,1), (2,3), (3,2), (4,3), (5,5)

Let **k = 2**.

#### Initial centroids:

* **C1 = (2,1)**
* **C2 = (2,3)**

#### Compute Distances (Iteration 1)

Use **Euclidean Distance Formula:**

$$
d(x,y) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}
$$

| Point | (x, y) | d(C1) | d(C2) | Assigned Cluster |
| :---- | :----- | ----: | ----: | :--------------: |
| P1    | (1,1)  |  1.00 |  2.24 |      **C1**      |
| P2    | (2,1)  |  0.00 |  2.00 |      **C1**      |
| P3    | (2,3)  |  2.00 |  0.00 |      **C2**      |
| P4    | (3,2)  |  1.41 |  1.41 |      **C1**      |
| P5    | (4,3)  |  2.83 |  2.00 |      **C2**      |
| P6    | (5,5)  |  5.00 |  3.61 |      **C2**      |


#### **Clusters After Iteration 1**

* **Cluster 1 (C1):** P1(1,1), P2(2,1), P4(3,2)
* **Cluster 2 (C2):** P3(2,3), P5(4,3), P6(5,5)

#### Compute New Centroids

$$
\text{Centroid} = \left(\frac{\sum x}{n}, \frac{\sum y}{n}\right)
$$


* $$C1= \frac{1}{3} ((1,1)+ (2,1)+ (3,2))=  (2.0, 1.33) $$
* $$C2=\frac{1}{3} ( (2,3)+ (4,3)+ (5,5))=   (3.67, 3.67)   $$

#### Recalculate Distances (Iteration 2)

| Point | (x, y) | d(C1:2.0,1.33) | d(C2:3.67, 3.67) | Assigned Cluster |
| :---- | :----- | -------------: | ------------: | :--------------: |
| P1    | (1,1)  |           1.05 |          3.78 |      **C1**      |
| P2    | (2,1)  |           0.33 |          3.15 |      **C1**      |
| P3    | (2,3)  |           1.67 |          1.8 |      **C1**      |
| P4    | (3,2)  |           1.204 |          1.8|      **C1**      |
| P5    | (4,3)  |           2.605 |          0.75 |      **C2**      |
| P6    | (5,5)  |           4.74 |          1.88 |      **C2**      |

#### **Clusters After Iteration 2**

* **Cluster 1 (C1):** P1(1,1), P2(2,1),P3(2,3), P4(3,2)
* **Cluster 2 (C2):** P5(4,3), P6(5,5)

#### Compute New Centroids


* $$C1= \frac{1}{4} ((1,1)+ (2,1)+(2,3)+ (3,2))=  (2.0, 1.75) $$
* $$C2=\frac{1}{2} ((4,3)+ (5,5))=   (4.5, 4)   $$

#### Recalculate Distances (Iteration 3)

| Point | (x, y) | d(C1:2.0,1.75) | d(C2:4.5, 4) | Assigned Cluster |
| :---- | :----- | -------------: | ------------: | :--------------: |
| P1    | (1,1)  |           1.25 |          4.61 |      **C1**      |
| P2    | (2,1)  |           0.75 |          3.9 |      **C1**      |
| P3    | (2,3)  |           1.25 |          2.69 |      **C1**      |
| P4    | (3,2)  |           1.03 |          2.5|      **C1**      |
| P5    | (4,3)  |           2.36 |          1.12 |      **C2**      |
| P6    | (5,5)  |           4.42 |          1.12 |      **C2**      |

#### **Clusters After Iteration 3**

* **Cluster 1 (C1):** P1(1,1), P2(2,1),P3(2,3), P4(3,2)
* **Cluster 2 (C2):** P5(4,3), P6(5,5)
  
✅ No change in cluster assignments → **Algorithm converged.**

---

### Problem 2

Cluster the points below into **k = 2** clusters using K-Means (Euclidean distance).

Points:

* P1 = (2,3)
* P2 = (5,6)
* P3 = (8,7)
* P4 = (1,4)
* P5 = (2,2)
* P6 = (6,7)
* P7 = (3,4)
* P8 = (8,6)

Initial centroids:

* **C1 = (2,3)**
* **C2 = (5,6)**


#### The distances to initial centroids and assignment

Use Euclidean distance $d((x_1,y_1),(x_2,y_2)) = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$.

| Point | Coordinates | d to C1=(2,3) | d to C2=(5,6) | Assigned |
| ----- | ----------: | ------------: | ------------: | :------: |
| P1    |       (2,3) |          0.00 |          4.24 |  **C1**  |
| P2    |       (5,6) |          4.24 |          0.00 |  **C2**  |
| P3    |       (8,7) |          7.21 |          3.16 |  **C2**  |
| P4    |       (1,4) |          1.41 |          4.47 |  **C1**  |
| P5    |       (2,2) |          1.00 |          5.00 |  **C1**  |
| P6    |       (6,7) |          5.66 |          1.41 |  **C2**  |
| P7    |       (3,4) |          1.41 |          2.82 |  **C1**  |
| P8    |       (8,6) |          6.70 |          3.00 |  **C2**  |

#### **Clusters after Iteration 1**

* **Cluster 1 (C1):** P1 (2,3), P4 (1,4), P5 (2,2), P7 (3,4)
* **Cluster 2 (C2):** P2 (5,6), P3 (8,7), P6 (6,7), P8 (8,6)


#### Compute new centroids (after Iteration 1)

Centroid = mean of coordinates of points in the cluster.

$$
\text{Centroid 1} = \text{mean of {(2,3),(1,4),(2,2),(3,4)}} = (\frac{2+1+2+3}{4} , \frac{3+4+2+4}{4}  )=( 2.0, 3.25 )
$$

$$
\text{Centroid 1} = \text{mean of {(5,6),(8,7),(6,7),(8,6)}} = 
  ( \frac{5+8+6+8}{4} , \frac{6+7+7+6}{4} )= (6.75, 6.50)$$


#### Iteration 2 — distances to updated centroids and assignment

| Point | Coordinates | d to C1_new=(2.00,3.25) | d to C2_new=(6.75,6.50) | Assigned |
| ----- | ----------: | ----------------------: | ----------------------: | :------: |
| P1    |       (2,3) |                    0.25 |                    5.90 |  **C1**  |
| P2    |       (5,6) |                    4.07 |                    1.82 |  **C2**  |
| P3    |       (8,7) |                    7.08 |                    1.35 |  **C2**  |
| P4    |       (1,4) |                    1.25 |                    5.77 |  **C1**  |
| P5    |       (2,2) |                    1.11 |                    6.54 |  **C1**  |
| P6    |       (6,7) |                    5.48 |                    0.2 |  **C2**  |
| P7    |       (3,4) |                    12.48 |                    4.55 |  **C1**  |
| P8    |       (8,6) |                    6.60 |                    1.35 |  **C2**  |


#### **Clusters after Iteration 1**

* **Cluster 1 (C1):** P1 (2,3), P4 (1,4), P5 (2,2), P7 (3,4)
* **Cluster 2 (C2):** P2 (5,6), P3 (8,7), P6 (6,7), P8 (8,6)

Assigned clusters did not change from Iteration 1 → the algorithm has **converged**.

---

# Hierarchical-Clustering-Analysis


**Hierarchical Cluster Analysis (HCA)** is an **unsupervised learning technique** that groups data into a hierarchy of clusters.  
Unlike partitioning methods such as K-Means, HCA **does not require specifying the number of clusters (k)** in advance.

---

## Definition

Hierarchical clustering is a method of cluster analysis that seeks to build a hierarchy of clusters by either:
- **Agglomerative (bottom-up)** merging of smaller clusters, or
- **Divisive (top-down)** splitting of larger clusters.

Clusters are formed based on **distance or similarity** between data points.

Represented using a **dendrogram**:
A dendrogram is like a family tree for clusters. It shows how individual data points or groups of data merge together.

---

## Steps in Hierarchical Clustering

1. Compute the **distance matrix** between all pairs of objects.
2. Each object starts as a **single cluster**.
3. Merge the **two closest clusters** based on a chosen **linkage criterion**.
4. Update the distance matrix to reflect the new merged cluster.
5. Repeat steps 3–4 until all objects are grouped into a single cluster.

---

## Approaches of HCA

| Approach | Description | Method |
|-----------|--------------|---------|
| **Agglomerative (Bottom-Up)** | Start with each point as a single cluster and recursively merge the nearest clusters until one cluster remains. | **AGNES (Agglomerative Nesting)** |
| **Divisive (Top-Down)** | Start with all data points in a single cluster and recursively divide it into smaller clusters. | **DIANA (Divisive Analysis)** |
<img width="660" height="360" alt="image" src="https://github.com/user-attachments/assets/7f03c7ca-b580-4f5f-9ccd-9c096949b880" />


---

## **Algorithm 1: Agglomerative Hierarchical Clustering (AGNES)**

Agglomerative Hierarchical Clustering (also known as **AGNES**-Agglomerative Nesting ) is a **bottom-up** approach:
Each data point starts as its own cluster, and clusters are **iteratively merged** based on similarity (or distance) until one single cluster remains.


### **Algorithm Steps**

1. **Start** with *n* clusters, each containing one data point (singleton). 
2. **Compute** the pairwise **distance matrix** between all clusters.
3. **Find** the two clusters with the **minimum distance** (most similar).
4. **Merge** these two clusters into one new cluster.
5. Compute the distance between the new cluster and each of the old clusters.
6. **Update** the distance matrix to reflect the new distances between clusters, using a chosen **linkage method**:

   * *Single linkage:* minimum distance
   * *Complete linkage:* maximum distance
   * *Average linkage:* mean distance
7. **Repeat** steps 3–5 until all points are merged into a single cluster.
8. The process can be visualized as a **dendrogram**, showing how clusters are merged step by step.

---

## **Algorithm 2: Divisive Hierarchical Clustering (DIANA)**

Divisive Hierarchical Clustering (also known as **DIANA**-Divisive Analysis) is a **top-down** approach:
Start with **one large cluster** containing all data points, and then **split** clusters recursively into smaller clusters until each data point stands alone.


### **Algorithm Steps**

1. **Start** with all data points in one cluster.
2. **Compute** the dissimilarity (distance) between all objects in the cluster.
3. **Find** the most dissimilar object — the point farthest from all others.
4. **Use** this object as the **seed** of a new cluster.
5. **Assign** each remaining point to one of the two clusters:

   * If it is **closer to the seed**, move it to the new cluster.
   * Otherwise, keep it in the existing cluster.
6. **Recalculate** dissimilarities within each cluster.
7. **Select** the cluster with the **largest average dissimilarity** and **repeat the split**.
8. **Continue** until every data point forms its own cluster or until the desired number of clusters is reached.

---


## Types of Linkage Methods

Linkage defines how the distance between two clusters is computed.

| Method | Description | Formula |
|---------|--------------|----------|
| **Single Linkage (Nearest Neighbor)** | Minimum distance between any two points in two clusters. | $D(C_1, C_2) = \min_{i \in C_1, j \in C_2} d(i, j)$ |
| **Complete Linkage (Farthest Neighbor)** | Maximum distance between any two points in two clusters. | $D(C_1, C_2) = \max_{i \in C_1, j \in C_2} d(i, j)$ |
| **Average Linkage** | Average distance between all points in the two clusters. |  |

![image](https://github.com/user-attachments/assets/a3f77b15-c862-4ee9-ad5e-e9d9e767f5b1)

---


## Example 

### Problem 1 

Distance Matrix

|   | P1 | P2 | P3 | P4 | P5 |
|:-:|---:|---:|---:|---:|---:|
| **P1** | 0 |   |   |   |   |
| **P2** | 9 | 0 |   |   |   |
| **P3** | 3 | 7 | 0 |   |   |
| **P4** | 6 | 5 | 9 | 0 |   |
| **P5** | 11 | 10 | 2 | 8 | 0 |

Perform clustering using **Complete Linkage Method**.


#####  **Start** with *5* clusters, each containing one data point (singleton). **Given Distance Matrix**

|        | {P1} | {P2} | {P3} | {P4} | {P5} |
| :----: | -: | -: | -: | -: | -: |
| {**P1**} |  0 |  9 |  3 |  6 | 11 |
| {**P2**} |  9 |  0 |  7 |  5 | 10 |
| {**P3**} |  3 |  7 |  0 |  9 |  2 |
| {**P4**} |  6 |  5 |  9 |  0 |  8 |
| {**P5**} | 11 | 10 |  2 |  8 |  0 |


##### Step 1 — Identify the smallest distance

We look for the **minimum non-zero value** in the matrix.

* Minimum = **2**, between **P3 and P5**.

**Merge C1 = {P3, P5}**


##### Step 2 — Recompute the distance matrix

Since we are using **Complete Linkage**, the new distance between a merged cluster and another cluster is the **maximum** of pairwise distances.

$$
D(C_{new}, X) = \max(D(P3,X), D(P5,X))
$$

* $d({P3, P5},P1)=max({d(P3,P1),d(P5,P1)})=max(3,11)=11$
* $d({P3, P5},P2)=max({d(P3,P2),d(P5,P2)})=max(7,10)=10$
* $d({P3, P5},P4)=max({d(P3,P4),d(P5,P4)})=max(9,8)=9$

So the new distance matrix becomes:

|             | {P1} | {P2} | {P4} | {P3,P5} |
| :---------: | -: | -: | -: | ------: |
|    {**P1**}   |  0 |  9 |  6 |      11 |
|    {**P2** }  |  9 |  0 |  5 |      10 |
|    {**P4**}   |  6 |  5 |  0 |       9 |
| {**P3,P5**} | 11 | 10 |  9 |       0 |


##### Step 3 — Find the next minimum distance

From the updated matrix,
the smallest value = **5**, between **P2 and P4**.

**Merge C2 = {P2, P4}**


##### Step 4 — Recompute the distance matrix again

Now clusters are:

* **C1 = {P3,P5}**
* **C2 = {P2,P4}**
* **P1**

We update distances using **maximum distance rule** (complete linkage).

* $d({P2,P4},P1)=max({d(P2,P1),d(P4,P1)})=max({9,6})=9$
* $d({P2,P4},{P3,P5})=max({d(P2,{P3,P5}),d(P4,{P3,P5})})=max({d(P3,P2),d(P5,P2),d(P3,P4),d(P5,P4)}=max{7,10,9,8})=10$


|             | P1 | (P2,P4) | (P3,P5) |
| :---------: | -: | ------: | ------: |
|    **P1**   |  0 |       9 |      11 |
| **(P2,P4)** |  9 |       0 |      10 |
| **(P3,P5)** | 11 |      10 |       0 |


##### Step 5 — Find the next smallest distance

The smallest distance now is **9**, between **P1 and (P2,P4)**.

**Merge C3 = {P1, P2, P4}**


##### Step 6 — Recompute distance matrix

Clusters now:

* **C1 = {P3,P5}**
* **C3 = {P1,P2,P4}**

Compute the distance between the two clusters using the **maximum** pairwise distance.

* $d({P1,P2,P4},{P3,P5})=max({d(P1,{P3,P5}),d({P2,P4},{P3,P5})})=max({d(P3,P1),d(P1,P5),d(P3,P2),d(P2,P5),d(P3,P4),d(P5,P4)})= \max(3, 7, 9, 11, 10, 8) = 11$

So the final merge distance = **11**.


#### Step 7 — Construct the Dendrogram

| Merge Step | Clusters Merged      | Distance (Level) |
| ---------- | -------------------- | ---------------- |
| 1          | (P3, P5)             | 2                |
| 2          | (P2, P4)             | 5                |
| 3          | (P1, P2, P4)         | 9                |
| 4          | (P1, P2, P3, P4, P5) | 11               |

<img width="686" height="474" alt="image" src="https://github.com/user-attachments/assets/ba68ceca-ad9f-4a8d-8b8d-5ee13efddcea" />



##### **Final Results**

* **Number of iterations:** 4
* **Final cluster formed at distance = 11**
* **Final 2 clusters before full merge:**

  * **Cluster A:** {P1, P2, P4}
  * **Cluster B:** {P3, P5}



📺 **Reference Video:** [Complete Linkage HCA Example](https://www.youtube.com/watch?v=JeY9P-Vw9hg)

---
### Problem 2

Create dendrogram using **Single Linkage Method** (Agglomerative Clustering)”

and the given **distance matrix**:

|        | P1 | P2 | P3 | P4 | P5 |
| :----: | -: | -: | -: | -: | -: |
| **P1** |  0 |  9 |  3 |  6 | 11 |
| **P2** |  9 |  0 |  7 |  5 | 10 |
| **P3** |  3 |  7 |  0 |  9 |  2 |
| **P4** |  6 |  5 |  9 |  0 |  8 |
| **P5** | 11 | 10 |  2 |  8 |  0 |



In **Single Linkage**,
the distance between two clusters is the **minimum distance** between any two points in the clusters.

$$
D(C_1, C_2) = \min_{i \in C_1, j \in C_2} d(i,j)
$$

##### Step 1 — Identify the smallest distance

From the matrix, the **smallest non-zero distance = 2**, between **P3 and P5**.

Merge: **C1 = {P3, P5}**


##### Step 2 — Update the distance matrix

Use the **minimum** distance rule to find distances from C1 to all other points.

* $d(({P3, P5}),P1)=min({d(P3,P1),d(P5,P1)})=min(3,11)=3$
* $d(({P3, P5}),P2)=min({d(P3,P2),d(P5,P2)})=min(7,10)=7$
* $d(({P3, P5}),P4)=min({d(P3,P4),d(P5,P4)})=min(9,8)=8$

New matrix:

|             | P1 | P2 | P4 | (P3,P5) |
| :---------: | -: | -: | -: | ------: |
|    **P1**   |  0 |  9 |  6 |       3 |
|    **P2**   |  9 |  0 |  5 |       7 |
|    **P4**   |  6 |  5 |  0 |       8 |
| **(P3,P5)** |  3 |  7 |  8 |       0 |


##### Step 3 — Find next minimum distance

Smallest = **3**, between **P1** and **(P3,P5)**.

Merge: **C2 = {P1, P3, P5}**



##### Step 4 — Update distance matrix

Now clusters are:

* **C2 = {P1, P3, P5}**
* **P2**
* **P4**

Compute new distances using **minimum** linkage rule:

* $d(({P1, P3, P5}), P2) = \min(d(P1,P2), d(P3,P2), d(P5,P2)) = \min(9,7,10) = 7$
* $d(({P1, P3, P5}), P4) = \min(d(P1,P4), d(P3,P4), d(P5,P4)) = \min(6,9,8) = 6$
* $d(P2,P4) = 5$


|                | (P1,P3,P5) | P2 | P4 |
| :------------: | ---------: | -: | -: |
| **(P1,P3,P5)** |          0 |  7 |  6 |
|     **P2**     |          7 |  0 |  5 |
|     **P4**     |          6 |  5 |  0 |



#####  Step 5 — Next smallest distance

Minimum = **5**, between **P2 and P4**.

Merge: **C3 = {P2, P4}**


#####  Step 6 — Update matrix

Clusters now:

* **C2 = {P1, P3, P5}**
* **C3 = {P2, P4}**

Compute new distances using **minimum** linkage rule:
* $d(({P1, P3, P5}), ({P2, P4})) = \min(d(P1,P2), d(P1,P4), d(P3,P2), d(P3,P4), d(P5,P2), d(P5,P4))= \min(9,6,7,9,10,8) = 6$

Final matrix:

|                | (P1,P3,P5) | (P2,P4) |
| :------------: | ---------: | ------: |
| **(P1,P3,P5)** |          0 |       6 |
|   **(P2,P4)**  |          6 |       0 |


#####  Step 7 — Final merge

Merge the last two clusters at **distance = 6**



**Final Two Clusters (Before Final Merge):**

* **Cluster 1:** {P1, P3, P5}
* **Cluster 2:** {P2, P4}

<img width="678" height="474" alt="image" src="https://github.com/user-attachments/assets/c2709569-fcfa-4f1d-b425-6b87030b38cd" />

---

## Advantages of Hierarchical Clustering

* No need to specify number of clusters (k) in advance.
* Produces a **hierarchical structure** that’s easy to interpret.
* Works well with small datasets and **non-spherical** clusters.
* **Flexible distance metrics** supported.

---

## Disadvantages of Hierarchical Clustering

* **Computationally expensive** (O(n³) time complexity).
* **Not scalable** to very large datasets.
* **Sensitive to noise/outliers**.
* **Irreversible:** Once clusters are merged, they can’t be undone.

---

## Applications

| Domain               | Application                             |
| -------------------- | --------------------------------------- |
| **Marketing**        | Customer segmentation                   |
| **Healthcare**       | Grouping patients with similar symptoms |
| **Education**        | Grouping students based on performance  |
| **Image Processing** | Identifying similar image regions       |
| **Finance**          | Portfolio or stock grouping             |

---

## Comparison – HCA vs K-Means

| Feature            | Hierarchical Clustering | K-Means Clustering        |
| ------------------ | ----------------------- | ------------------------- |
| Cluster type       | Different partiitions   | Single partition          |
| Number of clusters | Not needed initially    | Must specify (k)          |
| Scalability        | Low                     | High                      |
| Output             | Dendrogram              | Centroids                 |
| Runtime Efficiency | Less                    | More                      |

---
