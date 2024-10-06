# Classification
As said before the classification method can be divided into two macrogroups: **Unsupervised and Supervised**.

The **Unsupervised** mining technique are usually known in literature with a different name from classification, in this course when talking about classification we refer to **Supervised Classification**.

Let's consider the **Soybean**.

Here in this case the data set X contains N *individuals* described by D attributes value each.

Also we have a Y vector which, for each individual x contains the class value y(x).
The class allows a finite set of different values C.
The value of the class are provided by experts or the supervisors.

Our goal is to learn how to guess the value of the y(x) for individuals which have not been examined by the experts also called a *classification model*.

## Classification Model
A classification model is an algorith which, given an individual for which the class is not known, it compute the class.
The goal is to optimize the result of the algorithm.
THe main requirements to develop an algorithm like this are:
- The learning algorithm
- Use it to understand the paremetrization
- Assest the quality of the classification model

A decision function represent the the score for each raw data based represented on the prediction plane as: \

$M(\mathbf{x}, \theta_q) = y(\mathbf{x})_{\text{pred}}$

Where $y(x)pred$ is the prediction for the raw value $x$ and $M(x,\theta_q)$ is the model's prediction based on the input $x$ and the learning parameter $\theta_q$.

As we can see in the prediction the value under the line(hyperplane) are classified as class 2 and the other classify as 1.

![](/Theory/Images/Classification.png)

Here the decision function is a straigth line, but sometimes it may be a more complex function.

$\theta_1*d_1+\theta_2*d+\theta_3\geq 0$ is 2

$\theta_1*d_1+\theta_2*d+\theta_3<0$ is 1

The $\theta_i$ are the parameters.

All the models, even the best one produces are errors sometimes. Obvioulsy, the more parameter we have, the more precise is the model.

## Vapnik-Chervonenkis Dimension

Given a dataset with N elements there are 2 to the N possible different learning problems, if the model $M$ is able to shatter all the possible learning problems with N elements it is called *Vapnik-Chervonenkis Dimension* equal to N.
The line used before has a VC dimension of 3.

## The Classification Workflow

The phases of the creation of a model for classification are mostly the same everytime.

### Learning the model for a given set of classes
First a training set is necessary, containing a number of individuals necessary. 
Each individual hase the value of the class (which is named **Ground Thruth**).
The training set should be as much possible similar to a real one. 

### Estimate the accurancy of the model

Once the model is created is tested on a *test set* for which the class label for the individuals are known.
The model is run on the test set to decide the labels for the individual's classes and then is compared with the real one to measure the accurancy.

The workflow follow this schema most of the time:
![](/Theory/Images/ClassificationWorkFlow.png)

The classification is splitted into two **flavors**.

- Crisp: the classifiers assign to each individual one label
- Probabilistic: the classifier assign a probability for each of the possible labels to the individual

## Decision Trees
Decision trees are among the most used tools, they are useful to generate classifiers structured as *trees*.

The decision tree's structure is composed by *inner nodes* and *leaf nodes*.

The inner nodes are those where the decision is made, like the "if" statement, while the leaf node are those who actually predicts the class of the element $x$.

## Evaluate how much a pattern is interesting

There are several method, one of the is based on the information theory. In particular the well known **entropy**.

When we have an high entropy it means that the probability are mostly similar, thus the histogram is mostly flat, which is not so useful, instead if the entrpy is low, it means that some symbol are much more probable than others. 

A possible option to reduce the entropy and increase the purity of the set is to use a tecnique called **Threshold-based Split**, where the dataset based on whether a feature value exceeds a certain threshold $t$. 

A possible example: if the feature is the "age", it is possible to split the data into two groups: one where the age is greater than $t$, and the other where it is less than $t$.

After the split the **weigthed average entropy** for the two subset can be calculated and compared with the original one. The difference is called *Information Gain* or *IG*.

If the threshold is good the entropy deminish after the split.

A possible solution for avoiding higher entropy is to choose correctly the parameter on which test the set.

To test the method and evaluate it, most of the time is necessary to use the original dataset.
In particulare the test is splitted into:

- Training Set 
- Test Set

The Training set will be used to train the model, while the test set instead will be used to evaluate the learned model.

The split must be done randomly, because we consider as assumption that the parts have similar charateristics.

Once the split is done it is possible to calculate the information gain, this task should be performed on the attribute that gives the highest information gain. 

The split can be further applied more and more, thus creating a tree. Each subset can be split more and more.

The final conditions for the split, where after these is not possible to split anymore are: 
- Pure Nodes: All the data points in the node belong to the same class
- No Feature Left: No remaining features to split on are present
- Minimum Samples: The node has too few samples to split further
- Maximum Depth Reached: The tree has reached its maximum allowable depth, so further splits are restricted
- No information Gain: the split does not improve the IG


Some observation are useful 
to understand how it works:

- The weigthed sum of the entropy is always smaller than the ancestor one.


A useful value to understand if the model is well trained is the **Training Set Error**.

It is obtained by comparing the number of wrongly predicted class on the total predicted class on the training set itself.

It can be different from zero if the information are insufficient or the tree fails.
It defines the error we make on the data we used to generate the classification model.

A more accurate value to test the the expected behaviour with new data is the **Test set error**.

Most of the time it is way worst than the **Training set error**.
The cause may be **Overfitting**, which happens when the learning is affected by *noise*.

And this happens because a decision tree is an hypothesis of the relationship between the predictor attributes and the class. 

if a *hypotesis* does not perform well on the whole test set, but on a small size of it, it means that it is **Overfitting** the set.

The cause of overfitting are usually:
- Presence of Noise
- Lack of representative instances

This interpretation follows the Occam's razor law: \

$ Everything $ $should$ $be$ $made$ $as$ $simple$ $as$ $possible$, $but$ $not$ $simpler$

Meaning that in situations where the things are being equal, the simplest theory is preferable

If a long hypothesis fits the data, is more likely to be a coincidence.

A possible way to simplify this problem is pruning.

# The impurity functions

All the possible ways to measure the impurity of a node are:

- Entropy
- Gini Index
- Misclassification Error

## Gini Index

The **Gini index** (also called **Gini impurity**) is a metric used in decision trees to measure how "pure" a split is. It helps determine the quality of a split by evaluating the distribution of classes within a node. The goal is to minimize the Gini index at each split, which means creating nodes where each class is as pure as possible (i.e., containing mostly one class).

### Formula:
The Gini index for a node is calculated as:

$
Gini = 1 - \sum_{i=1}^{C} p_i^2
$

Where:
- $ p_i $ is the proportion of class $ i $ in the node (the fraction of samples of class $ i $ compared to the total samples in the node).
- $ C $ is the total number of classes.

### How it works:
- **Gini index = 0**: The node is pure, meaning all the samples in that node belong to one class.
- **Gini index = 1**: The node is completely impure, with all classes being equally represented.

### Example:
For a binary classification problem:
- If a node contains 90% of samples from Class A and 10% from Class B, the Gini index will be low (closer to 0).
- If the node contains 50% from Class A and 50% from Class B, the Gini index will be higher (closer to 0.5).

### Use in Decision Trees:
At each decision point, the Gini index is used to evaluate possible splits. The split that minimizes the Gini index (i.e., creates the purest subgroups) is chosen.

## Misclassification Error

The **misclassification error** (or **classification error rate**) measures the proportion of incorrect predictions made by a classification model. It quantifies how often the model's predicted class label differs from the actual class label.

### Formula:
The misclassification error is calculated as:

$
\text{Misclassification Error} = \frac{\text{Number of Incorrect Predictions}}{\text{Total Number of Predictions}}
$

Alternatively, it can be written as:

$
\text{Misclassification Error} = 1 - \text{Accuracy}
$

Where **accuracy** is the proportion of correct predictions.

### Example:
- If a model makes 100 predictions and 10 of them are wrong, the misclassification error would be:
  $
  \frac{10}{100} = 0.1 \text{ or } 10\%
  $

![](/Theory/Images/ComparisonImpurityFunctions.png)

# Decision Tree Induction

The DT induction is the process of building a decision tree model from a given dataset. It involves the splitting of the data recursively based on the features and selecting the best feature at each step to partition the data. 

# Important Concepts 
- Impurity functions: entropy, Gini, misclassification
- The recursive greedy algorithm for building a decision tree
- Training error and test error
- Why the test error can be much greater than the training error
- Why the pruning can improve the performance
- How to deal with continuous attributes