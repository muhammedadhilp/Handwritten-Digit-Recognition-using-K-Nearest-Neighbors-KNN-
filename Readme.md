# Handwritten Digit Recognition using K-Nearest Neighbors (KNN)

## Introduction

This project aims to classify handwritten digits using the K-Nearest Neighbors (KNN) algorithm. The dataset used is the **Digits Dataset** (similar to MNIST), which contains 8x8 pixel grayscale images of handwritten digits (0-9). The goal is to preprocess the data, apply normalization, and train a KNN model to classify the digits.

## Dataset Description

### Dataset: Digits Dataset
- The dataset consists of 1,797 samples, each representing an 8x8 pixel image.
- Each sample has:
  - **Features**: 64 numerical features representing the grayscale intensity (from 0 to 16) of each pixel in the image.
  - **Target**: A numerical label representing the digit (0-9) the image corresponds to.

### Loading the Dataset
We load the dataset from `sklearn`:
```python
from sklearn import datasets
digits = datasets.load_digits()
```

### Preprocessing the Data
Train-Test Split
The dataset is split into 80% training data and 20% testing data:

```python

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
```
##Normalization
Since the pixel values range from 0 to 16, normalization is essential to bring these values into a standard range. We chose Min-Max Scaling to normalize the data between 0 and 1:

```python

x_train = x_train / 16.0
x_test = x_test / 16.0
```
### Normalization helps:

- Consistent Scale: Prevents features with larger scales from dominating the model.
- Improved Model Performance: Algorithms like KNN benefit from having features on a consistent scale, as KNN relies on distance metrics.
- Model Building: K-Nearest Neighbors (KNN)
## KNN Classifier
We initialize the KNN model with k=3 neighbors. KNN works by finding the nearest neighbors to a data point and classifying it based on the majority class of its neighbors.

```python

from sklearn.neighbors import KNeighborsClassifier
k_value = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k_value)
knn_classifier.fit(x_train, y_train)
```
## Model Prediction
Once the KNN model is trained, we use it to predict the labels for the test set:

```python

y_pred = knn_classifier.predict(x_test)
```
## Model Evaluation
We evaluate the model using common classification metrics such as accuracy, precision, recall, and F1 score:

```python

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
```
### Results
- Accuracy: Measures the percentage of correctly classified digits.
- Precision: How many selected items are relevant.
- Recall: How many relevant items are selected.
- F1 Score: A harmonic mean of precision and recall, representing the balance between the two.
For example, if the evaluation metrics print:

```yaml

Accuracy: 0.9917
Precision: 0.9917
Recall: 0.9917
F1 Score: 0.9917
This means the model is performing with 99% accuracy on the test data.
```
## Visualization of Predictions
We visualize the first five test images along with their predicted labels:

```python

import matplotlib.pyplot as plt
new_digit_predictions = knn_classifier.predict(x_test[:5])

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f'Predicted: {new_digit_predictions[i]}')
    plt.axis('off')

plt.show()
```
## Conclusion
This project demonstrates how to use the KNN algorithm to classify handwritten digits. The model achieved high accuracy, and we used normalization to ensure the features are on a consistent scale for better performance. The results show that the KNN model is well-suited for this type of image classification task.

## Requirements
To run this project, you'll need the following Python libraries:

- numpy
- pandas
- scikit-learn
- matplotlib
You can install them using the following command:

```bash
pip install numpy pandas scikit-learn matplotlib
```
## Future Work
Experiment with different values of k to improve model performance.
Try different normalization techniques (e.g., standardization) and compare results.
Implement other classification models such as Support Vector Machines (SVM) or Convolutional Neural Networks (CNN) for better accuracy on handwritten digit classification.
