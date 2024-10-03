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

