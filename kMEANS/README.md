# K-Means Clustering From Scratch

A Python implementation of the K-Means Clustering algorithm built from scratch using NumPy. This project demonstrates how K-Means works internally without relying on machine learning libraries such as Scikit-Learn.

## Overview

K-Means is an unsupervised machine learning algorithm used to group similar data points into clusters. The algorithm iteratively assigns data points to the nearest centroid and updates centroid positions until convergence.

This implementation includes:

- Random centroid initialization
- Euclidean distance calculation
- Cluster assignment
- Centroid updates
- Convergence checking
- Cluster visualization using Matplotlib

## Technologies Used

- Python
- NumPy
- Matplotlib

## Algorithm Steps

1. Initialize `k` centroids randomly within the data range.
2. Calculate the distance between each data point and every centroid.
3. Assign each data point to the nearest centroid.
4. Recalculate centroids as the mean of all points assigned to them.
5. Repeat steps 2–4 until the centroids stop changing significantly or the maximum number of iterations is reached.

## Usage

Generate a random dataset and perform clustering:

```python
random_points = np.random.randint(0, 100, (100, 2))

kmeans = KMeansClustering(k=3)
labels = kmeans.fit(random_points)
```

Visualize the clusters:

```python
plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)

plt.scatter(
    kmeans.centroids[:, 0],
    kmeans.centroids[:, 1],
    marker="*",
    s=200
)

plt.show()
```

## Example Output

The algorithm clusters the data points into groups and displays:

- Colored points representing cluster assignments
- Star-shaped markers representing cluster centroids

## Complexity Analysis

| Operation | Time Complexity |
|------------|----------------|
| Distance Calculation | O(n × k) |
| Cluster Assignment | O(n × k) |
| Centroid Update | O(n) |
| Overall | O(n × k × iterations) |

Where:

- n = Number of data points
- k = Number of clusters
- iterations = Number of iterations until convergence

## Key Concepts Demonstrated

- Unsupervised Learning
- Clustering
- Euclidean Distance
- Iterative Optimization
- NumPy Vectorization

## Future Improvements

- K-Means++ Initialization
- Elbow Method
- Silhouette Score Evaluation
- Higher-Dimensional Data Support
- Animated Visualization of Clustering Process

## Author

Nithin

