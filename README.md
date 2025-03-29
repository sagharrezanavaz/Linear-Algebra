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




# Word Embeddings with GloVe and Co-occurrence Matrix

This notebook explores word embeddings using the pre-trained GloVe model and by creating embeddings from a co-occurrence matrix.

## 1. GloVe: Global Vectors for Word Representation

- **Exploring word vectors:** We examined how GloVe embeddings capture semantic relationships between words, such as gender and royalty.
- **Visualizing in 2-D:** We used PCA and t-SNE to reduce the dimensionality of GloVe embeddings and visualize relationships between words like "tall," "taller," and "tallest" in a 2D space. The visualizations showed how semantically similar words tend to cluster together in the reduced-dimensional space.

## 2. Evaluation

- **Cosine similarity:** We used cosine similarity to measure the semantic similarity between word vectors, such as "france" and "paris." We found that words with similar meanings have higher cosine similarity scores.
- **Analogy test:** We evaluated the GloVe model's ability to solve word analogies using a small test set. The model achieved an accuracy of approximately 0.6 (or 60%), indicating its ability to capture some analogical relationships between words.

## 3. Learn embeddings

- **Dataset Preprocessing:** We preprocessed a Simpsons dataset, cleaning the text data by removing stop words, lemmatizing words, and filtering tokens. This prepared the data for creating word embeddings.
- **Tokenization and Phrase Detection:** We tokenized the cleaned sentences and detected common bigrams using Gensim's Phrases and Phraser tools. This allowed us to capture multi-word expressions as single units in the embeddings.
- **Co-occurrence Matrix:** We built a co-occurrence matrix to analyze word relationships based on their co-occurrence within a context window. The matrix stored the frequency of word pairs appearing together within a specified window size.
- **SVD:** We applied Singular Value Decomposition (SVD) to reduce the dimensionality of the co-occurrence matrix, creating word embeddings. This reduced the size of the embeddings while preserving important semantic information.
- **Similarity:** We calculated cosine similarity between the learned embeddings to find similar and dissimilar words. We observed that the learned embeddings captured semantic relationships between words, as similar words had higher cosine similarity scores. We also explored word analogies using vector arithmetic, finding that the embeddings could to some extent represent analogical relationships.


## 4. Sentiment Analysis of MDB Movie Reviews 

## Summary

This notebook demonstrates the use of pre-trained word embeddings (GloVe) and the process of creating word embeddings from a co-occurrence matrix. It covers data preprocessing, tokenization, phrase detection, co-occurrence matrix construction, dimensionality reduction using SVD, and similarity analysis. The notebook shows how word embeddings can capture semantic relationships between words and be used for tasks like word similarity and analogy detection.

