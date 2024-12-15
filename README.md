# Part 1 
In this project, we implement a classification algorithm for the famous **Iris Flower Dataset**, learning key concepts in data manipulation, vectorized computations, and machine learning basics.

## Project Overview

This project introduces the **k-Nearest Neighbors (kNN)** algorithm, a simple yet powerful classification technique used in machine learning. Our goal is to predict the type of an iris flower (Setosa, Versicolor, or Virginica) based on its features:

- Sepal length
- Sepal width
- Petal length
- Petal width

We use the following datasets:

- **irises.npy**: Feature data for 120 samples.
- **types.npy**: Corresponding labels (0, 1, or 2 for Setosa, Versicolor, or Virginica).
- **new\_irises.npy**: Feature data for 30 test samples.
- **new\_types.npy**: Ground truth labels for test samples (used for accuracy evaluation).

## Key Concepts and Steps

### 1. Loading the Dataset

We begin by loading the datasets using NumPy.


### 2. Distance Calculation

To classify a new sample, we calculate its distance to every training sample using three methods:

#### a. **Two-Loops**

We use nested loops to calculate the Euclidean distance between test and training samples.

#### b. **One-Loop**

We eliminate the inner loop using NumPy's broadcasting feature.

#### c. **No Loops**

We fully leverage NumPy's capabilities to compute distances without loops.


### 3. Finding k-Nearest Neighbors

For each test sample, we find the indices of the **k closest training samples** using `np.argpartition`.


### 4. Predicting Labels

Using the k-nearest training samples, we determine the majority label (mode) for each test sample.

### 5. Accuracy Evaluation

Finally, we compare predicted labels with ground truth to calculate accuracy.


### Output:

The kNN classifier achieves **100% accuracy** on the test dataset.

## Results

This project demonstrates:

- Efficient data processing with NumPy.
- Basics of machine learning, including training and testing datasets.
- Hyperparameter tuning with the `k` value.




