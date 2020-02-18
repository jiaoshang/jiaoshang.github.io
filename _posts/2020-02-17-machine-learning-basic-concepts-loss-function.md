---
title: Machine Learning Basic Concepts - Loss Function
tags: machine_learning
layout: article
mathjax: true
article_header:
  type: overlay
  theme: dark
---

<!--more-->
//todo logloss
## What is loss function?
A loss function tells how good our current model is,
- A loss function is used to train your model. A metric is used to evaluate your model.
- A loss function is used during the learning process. A metric is used after the learning process

> The function we want to minimize or maximize is called the objective function or criterion. When we are minimizing it, we may also call it the cost function, loss function, or error function. 
> 
> -- Section 4.3 [Deep Learning, 2016](http://www.deeplearningbook.org/)

## Regression Loss Functions
### Mean Squared Error Loss (L2 Loss)

$$
MSE =  \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{n}
$$

where $\hat{y}$ is the prediction and $y$ is the ground truth. 

Mean square error measures the average of squared difference between predictions and actual value. The MSE loss function penalizes predictions which are far away from actual values by squaring. This property makes the MSE less robust to outliers. Therefore, it should not be used if our data is prone to many outliers.

### Mean Squared Logarithmic Error Loss

$$
MSLE = \frac{\sum_{i=1}^{n}(log(y_i + 1) - log(\hat{y}_i + 1))^2}{n}
$$

where $\hat{y}$ is the prediction and $y$ is the ground truth. 

MSLE is usually used when you don't want to penalize huge difference between the predictions and true values when both of them are huge numbers.  

### Mean Absolute Error Loss (L1 Loss)

$$
MAE =  \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{n}
$$

where $\hat{y}$ is the prediction and $y$ is the ground truth. 

Mean absolute error is measured as the average of sum of absolute differences between predictions and actual values. MAE is more robust to outliers since it does not make use of square. However, handling the absolute or modulus operator in mathematical equations is not easy.

### Huber Loss

## Binary Classification Loss Functions
### Binary Cross-Entropy
#### Information Theory Background

In the information theory, the information content,
$$
I(X) = -logP(X)
$$

Expected value,

$$
E[X] =  \sum_{i=1}^{n} x_ip_i
$$

Entropy is used to measure the uncertainty of an event,

$$
H(X) = E[I(X)] = E[log(P(X))] = -\sum_{i=1}^{n} p(x_i)log(p(x_i))
$$

Cross-entropy is commonly used to quantify the difference between two probability distributions. 𝑝(𝑥), the true distribution, and 𝑞(𝑥), the estimated distribution, 

$$
H(p,q) =  -\sum_{i=1}^{n} p(x_i)log(q(x_i))
$$

#### Binary Cross Entropy Loss

Binary cross entropy loss is,

$$
Binary Cross Entropy= - (y_ilog(\hat{y}_i) + (1 - y_i)log(1 - \hat{y}_i)) = \begin{cases}
log(\hat{y}_i), where \quad y_i = 1 \\
log(1 - \hat{y}_i), where \quad y_i = 0
\end{cases}
$$

where $y$ is the ground truth vector and $\hat{y}$ is the prediction.

Cross-entropy loss increases as the predicted probability diverges from the actual label. Cross entropy loss penalizes heavily the predictions that are confident but wrong.

### Hinge Loss (SVM Loss)

$$
Hinge= \sum_{i =1} ^{n} max(0, 1 - y_i\hat{y_i})
$$

where $\hat{y}$ is the prediction and $y$ is either 1 or -1. 

Hinge loss is used for binary classification where the target values are either -1 or 1.

The hinge loss function encourages classifying examples into the correct class, assigning more error when there is a difference in the label between the actual values and prediction. It is used for maximum margin classification, most notably for support vector machines.

### Squared Hinge Loss

$$
Squared Hinge = \sum_{i =1} ^{n} max(0, 1 - y_i\hat{y_i})^2
$$

where $\hat{y}$ is the prediction and $y$ is either 1 or -1. 

The squared hinge loss is a loss function used for “maximum margin” binary classification problems. It punishes larger errors more significantly than smaller errors.

## Multi-Class Classification Loss Functions
### Multi-Class Cross-Entropy Loss

$$
Multi Class Cross Entropy = - \sum_{i = 1} ^{C}t_i log(\hat{y_i})
$$

where $t$ is a binary indicator (0 or 1), it is 1 if the class label is the ground truth, $\hat{y_i}$ is the predicted score for this class label (Softmax/ Sigmoid score), C is the number of classes.

### Sparse Multiclass Cross-Entropy Loss

### Kullback Leibler Divergence Loss

multiclass hinge loss

$$
Multi Class Hinge Loss = \sum_{j \not ={y_i}} max(0, s_j - s_{y_i} + 1)
$$