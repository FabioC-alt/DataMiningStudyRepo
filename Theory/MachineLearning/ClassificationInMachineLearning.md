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

# Model Selection

In this section we are going to further explore the available models for classification.

The amount of data in the set is most of the time scarce, because we have to split them among:

- Train
- Validation
- Test

Obviously the more data we use, the best performance we should expect.

To avoid the overoptimistic situation we decide a **coincidence interval** to be assigned to the test error.

This most of the time depends from the empirical frequency and are due to noise in the data. 

Usually noise is assumed to have a normal distribution around the true probability

## Establishing the accurancy of a classifier

The error frequency is the simplest indicator of the quality of a classifier. 

It is to be considered that every machine learning algorithm has one or more parameters that influence its behaviour. These parameters are called *hyperparameters*.

There are two **testing strategies** that are used to get the most out of the supervised data set.

- Holdout
- Cross Validation

### Holdout

A typical value of the training/test ratio is 2/1 and the split should be as random as possible. 
The test set is used mostly to obtain an estimation of the performance measures with new data.

### Cross-Validation

The training set is randomly partitioned into $k$ subset. 
All the subset ecept one are used for training, the last one is used for testing.
Finally the final model is generated using the *entire training set*, while all the previous results are combined.

#### Bootstrap
It is a statistical sampling technique, where $N$ records are sampled with **replacement**.

## Performance measures of a classifier

### Binary Prediction

$success rate = accurancy$

$accurancy = {TP+TN}/N_{test}$

- TP: True Positive
- TN: True Negative
- $N_{test}$: All the predictions
The accurancy gives an initial feeling of the effectiveness of the classifier,but can heavily misleading when classes are imbalanced.

Other possible performances measures are:

$Precision$ = $\frac{TP}{(TP+FP)}$

$Recall$ = $\frac{TP}{(TP+FN)}$

$FN$: False Negative

$Specificity$ = $\frac{TN}{(TN+FP)}$

$F1-score$ = $2 \frac{(prec*rec)}{(prec+rec)}$

The F1-score is *always* interesting, because it has higher values when precision and recall are reasonably balanced.

## Confusion Matrix
The ***confusion matrix*** contains the number of test records of class $i$ and predicted as class $j$, while the diagonal are only the true prediction.

![](/Theory/Images/ConfusionMatrix.png)

## $k$ statistic
Evaluates the concordance between two classifications. 

$\kappa = \frac{P_o - P_e}{1 - P_e}$

$P_o $ is the probability of concordance
$P_e $ is the probability of random concordance

![](/Theory/Images/SpectrumofK.png)

## The cost of errors
Producing errors in predicting classes sometimes may be even more expensive than not producing them, so it is possible to force weigthed errors:

- Alterate the proportion of classes in the supervised data, duplicating the examples for which the classification error is higher
- Using a learning scheme that allows to add weights to the instances.

# Evaluation of probabilistic classifiers

Sometimes may be useful to predict the probability of a record in all the possible classes (*CRISP*) method instead of the simple prediction. To do so, is possible to use two different tecniques:

- binary
- multiclass

## Binary

Used to evaluate various scenarios, depending on the application.

![](/Theory/Images/LiftChart.png)

The straigth line plots the number of positives obtained form a random choiche of a sample of test data.

The curve plots the number of positives obtained drawing a fraction of the test data with decreasing probability

The larger the area between the two curves, the best the classification model.

## ROC Curve
It is the tradeoff between the hit rate and the false alarm in a noisy channel.

![](/Theory/Images/ROCCurve.png)

# Statistical Modeling

Here are presented some of the statistical model based on the Bayes's theorem.

## Bayes' Theorem:
Naive Bayes is a probabilistic classification algorithm based on Bayes' Theorem. It assumes that the features of the data are independent of each other, which is a "naive" assumption, hence the name. Despite this simplification, Naive Bayes often performs surprisingly well for various types of classification problems.


The formula is:

$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$

Where:
- $P(A|B)$ is the probability of class A given feature B (posterior probability).
- $P(B|A)$ is the likelihood, or the probability of feature B given class A.
- $P(A)$ is the prior probability of class A.
- $P(B)$ is the probability of feature B.

It has a clear semantic for learning probabilistic knowledge and excellent results in many cases.

The biggest criticity of this tecnique are related to the assumption which are simplistic but sometimes they may not be strong enough. 
When we have missing values is necessary to find a way to solve the holes, a possible solution is to use the smoothing which can reduce the overfit.

If we have numeric values is impossible to apply the method, thus to overcome these problems a gaussian distribution is used. 


# Linear Perceptron

A **linear perceptron** is a type of binary classifier that uses a linear decision boundary to classify input data into one of two classes. It is one of the simplest types of neural networks and serves as the building block for more complex models.

## Perceptron Structure:
- The perceptron takes in multiple input features (e.g., $ x_1, x_2, ..., x_n $) and assigns a weight to each one (e.g., $ w_1, w_2, ..., w_n $).
- It computes a weighted sum of the input features, plus a bias term $b$, and applies an activation function to determine the output.

## Perceptron Equation:
The output of a perceptron is determined by the following equation:

$
y = f(w \cdot x + b)
$

Where:
- $ w \cdot x $ is the dot product of the weight vector $ w $ and the input vector $ x $, which results in a scalar value.
- $ b $ is the bias term, which helps adjust the decision boundary.
- $ f $ is the activation function, often a step function that outputs either 1 or -1 (or 0, depending on the problem).

## Step Function:
In binary classification, the activation function is typically a step function:

$
f(z) =
\begin{cases} 
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0 
\end{cases}
$

Where $ z = w \cdot x + b $.

## Classification Process:
1. **Input**: The perceptron takes the input vector $x$ (the features of the data).
2. **Weighted Sum**: It computes the weighted sum of the inputs and adds the bias term.
3. **Activation**: The step function determines whether the output is 0 or 1 (the class label).
4. **Classification**: Based on the result of the activation function, the perceptron assigns the input to one of the two classes.

## Training the Perceptron:
The perceptron uses a simple learning rule to adjust its weights during training, called the **perceptron learning algorithm**:
- For each training sample, if the perceptron classifies the input incorrectly, the weights are updated using the following rule:

$
w_i \leftarrow w_i + \Delta w_i
$
$
\Delta w_i = \eta (y_{\text{true}} - y_{\text{pred}}) x_i
$

Where:
- $ \eta $ is the learning rate, a small positive constant.
- $ y_{\text{true}} $ is the actual label of the training sample.
- $ y_{\text{pred}} $ is the predicted label from the perceptron.
- $ x_i $ is the input feature corresponding to weight $ w_i $

## Limitations:
- The perceptron can only classify data that is **linearly separable** (i.e., data that can be separated by a straight line in 2D or a hyperplane in higher dimensions).
- It cannot solve more complex problems like XOR, which are not linearly separable.

# Support Vector Machines (SVM) for binary classification
If the data are not linearly separable this means that the boundary between classes is some type of hyper-surface more complex than a hyperplane. One possibility is abandone the linearity, but if we do so the method would soon become intractable and also would be extremely prone to overfitting.

The main ideas are based on the computational learning theory and focuses on *Optimization* rather than greedy search.

The key concepts are:
- The hyperplane: is the decision boundary that separates the two classes
- The margin: is the distance between the hyperplane and the closest data points from each class, the goal of the SVM is to maximize this margin. 
- The support vector: These are the data points that are closest to the hyperplane, they are critical in determining the position and orientation of the hyperplane.
- The linear SVM: if datas are separated from a straight line, the problem is called linearly separable, if not is called Non-linear SVM.

## COMPLETARE

# Neural Networks

The functioning of the human brain inspired the creation of computer based lookalike structure such as the perceptrons. A neurons is a signal processor wich is triggered each time a threshold is reached, the signal from one neuron to another is **weigthed**, the weigth changes over time, also due to *learning*.

The signals are modeled as real number, the threshold of the biological system is modeled as a mathematic function, in particular the **Sigmoid** function:

![](/Theory/Images/Sigmoid.png)

Most of the result with the linear perceptron were non satisfactory because of the linearity, in addition to the problem of separability.

To imrove the results of the neural network more layer are posed onto each other each of these layers creates a feed to change the networks output.

## Training a neural network
Training a neural network means to encode the weigths and decide the value of each of them.
It is also possible to establish the error of each value:

![](/Theory/Images/ErrorInNeural.png)

The learning models of a neural network may be devide into two main groups:

- Stochastic: each forward propagation is immediately followed by a weigth update, this step introduces some noise
- Batch: before changing a weigths there are many propagations 

The training can be considered complete after many rounds of learning over the entire set(epoch), if the weigths update result in minimal changes.

If the network is too complex, with too many layers there may be overfitting.

A technique used in many machine learning functions to improve the generalisation capabilities of a model is **Regularization**. Regularization corrects the loss function in order to smooth the fitting to the data.

# K Nearest Neigthbours Classifiers
Keeps all the training data, makes prediction by computing the similarity between the new samples and each training instance. Picks the K entries in the database which are the closest to the new data point.

# Loss function
The loss function is a mathematical function that quantifies the difference between the predicted values by a model and the actual values. It serves as a measure of how well or poorly a model is performing, guiding the learning process by helping the model minimize the errors during the training.

# The binary classifiers and the Multi-class classification
When dealing with multi-class classfication we need other methods. There are two ways to deal with multi-class algorithm.
- Transform the training algorithm and the model
- Use a set of binary classifiers and combine the result

## OVO Strategy
In One-vs-One strategy we consider all the possible pairs of classes and generate a binary classifiers for each pair, each binary problem consider only the examples of the two selected classes. During the prediction time a voting scheme is implemented and used to classify the result using the +1. 

## OVR Strategy
In One-vs-Rest strategy we consider C binary problems where class *c* is a positive example and all the others are negatives.
At the prediction time a voting scheme is applied. The difference between the two is that the OVR splits the multi-class problem into multiple binary classification problem.
In the OVO approach the classifiers creates binary classification problems between every possible pair of classes.

While the OVO require solving a higher number of problems the OVR tends to be intrinsically unbalanced.

# Ensable Methods
Using more tree to obtain the best solution.
Train a set of **base classifiers** and the final prediction is obtained taking the votes of the base classifiers, the ensable methods tends to perform better than a single classifier. The ensable methods are useful if the base classifiers are independent and the performance of the base classifier is better than random choice.

In ensable methods there are many way to manipulate the data: 
- Bagging: repeatedly samples with replacements according to a uniform probability distribution
- Boosting: iteratively changes the distribution of training examples so that the base classifier focus on examples which are hard to classify
- Adaboost: the importance of each base classifier depends on it error rate 

# Forest of Randomised Tree
A technique called perturb-and-combine is specifically designed for trees, a diverse set of classifiers is created by introducing randomness in the classifiers construction. The prediction of the ensable is given as the average prediction of the individual classifiers 

## Bias-vs-Variance Tradeoff
A bias is the simplifying assumption made by the model to make the target function easier to approach.
Variance is the amount that the estimate of the target function will change, give different training data.
The Bias-variance Tradeoff is place where our machine model performs between errors introduce by the bias and the variance.

The purpouse of this two elements is to simplify the assumption of the model, but they introduce randomness. To reduce the randomness is possible to use the Random Forest technique, which creates multiple trees and makes possible to obtain more result, but not all the solutions are necessary only the best one is considered.

### Boosting 
Boosting with ensamble learning is possible by using a different classifier and the weigths are modified iteratively according to classifier performance.
In particular a machine learning algorithm used for boosting is the **AdaBoosting**(*Adaptive Boosting*).
Adaboosting is an ensable method that combines several weak classifiers to form a strong one, by focusing more on the misclassified examples in each iteration.



