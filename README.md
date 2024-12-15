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

In the end, we utilize the classifier for image processing purposes, experimenting with different k values.

## Results

This project demonstrates:

- Efficient data processing with NumPy.
- Basics of machine learning, including training and testing datasets.
- Hyperparameter tuning with the `k` value.
- Image processing.

# Part 2
# Diamonds and Elements Data Visualization

## Overview
This project focuses on data visualization using Python libraries such as **Pandas**, **Matplotlib**, and **Seaborn**. It is divided into two parts:

1. **Diamonds Dataset Analysis**: Visualizing information about diamonds using bar plots and box plots.
2. **Unidro Elements Dataset Analysis**: Using heatmaps to study correlations between various elements in a mining dataset.

---

## Prerequisites
Ensure you have the following installed in your environment:

- Python 3.7 or higher
- Pandas
- Matplotlib
- Seaborn


---

## Dataset Descriptions

### Diamonds Dataset
- **File**: `diamonds.csv`
- **Columns**:
  - `carat`: Weight of the diamond
  - `cut`: Quality of the diamond cut
  - `color`: Color grade of the diamond
  - `clarity`: Clarity grade of the diamond
  - `depth`: Total depth percentage
  - `table`: Width of the diamond’s top
  - `price`: Price in US dollars
  - `x`, `y`, `z`: Diamond dimensions (length, width, height)

### Unidro Dataset
- **File**: `unidro_data.csv`
- **Columns**: Concentrations of 31 chemical elements measured in parts per million (ppm) for 1226 samples.

---

## 1: Diamonds Dataset Analysis

### 1. Bar Plot: Price by Cut and Color
#### Description:
- Visualizes the average price of diamonds for each cut, grouped by color.
- Includes confidence intervals to show price variability.


### Box Plot: Price by Clarity
#### Description:
- Highlights the distribution of diamond prices for each clarity grade.
- Useful for identifying outliers and understanding price spread.


## 2: Unidro Dataset Analysis

### Heatmap: Correlation Matrix
#### Description:
- Displays the correlation between chemical elements using a heatmap.
- Helps identify meaningful relationships between elements.


## File Structure
```
Part 2/
├── diamonds.csv         # Diamonds dataset
├── unidro_data.csv      # Unidro elements dataset
├── analysis.py          # Main Python script
└── README.md            # Project documentation
```

---


## Output
- **Bar Plot**: Average diamond price by cut and color.
- **Box Plot**: Distribution of diamond prices by clarity.
- **Heatmap**: Correlation between chemical elements.

---

