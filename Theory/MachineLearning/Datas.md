# Data in Machine Learning

The data are caratherized by some qualities such as: the type and the quality.

In order to ease the mining activities the data need to be pre-processed and transformed.

## Data Types
The data types can be subdivided into two macro-categories: the categorical and the numerical.

![](/Theory/Images/TypesOfData.png)

The allowed transformation in data types are different if the data is categorical or numerical.

## Data Quality
The most present problem in data quality is the **Noise**, the noise is the modification of original value, which can compromise the data itself.

Another problem is the presence of **Missing Values** and **Duplicated Values**.

# The Data - Pre-processing and dissimilarities

The main topics when talking about data are:
- Aggregation
- Sampling
- Dimensionality Reduction
- Feature subset selection
- Feature creation
- Discretization and Binarization attribute transformation

## Aggregation

The main goal is to combine two or more attributes into one.
The purpose is to reduce the number of attributes or objects, change the scale of the aggregation and obtain more stable data.

## Sampling

Sampling is a crucial step to investigate and obtain the final data analysis, they are necessary both for the **preliminary** phase and also for the **analysis** phase.

The sampler should concentrate its forces on sample *representative* data. A sample which is representative has approximately the same property or interests as the original set of data.

### Type of sampling
1. Simple random data: a random choice of an object with given probability
2. Replacement: repetitions of independent extraction of type 1
3. without replacement: the same of the 2, but here the extraction is no more independent
4. Stratified: the data are splitted into more partitions according to some criteria, then draw the random sample from each partition

### Sample size
The sample should be decided based on the techniques provided by the statistics, in particular based on the tradeoff between data reduction and precision

![](/Theory/Images/SampleSize.png)

The sampling should be done in order to have at least one element for each class(with replacement)

Having missing classes becomes relevant, for example, in a supervised dataset with a high number of different values of the target

If the number of data elements is not big enougth, it can be difficult to guarantee a stratified partitioning in tran/test split or in cross-validation split.

## Dimensionality
When the dimensionality is very high the occupation of the space become very sparse, the discrimination on the basis of the distance becomes uneffective.
In order to reduce the dimensionality is possible to apply some techniques:

### Principal Components Analysis
Here the goal is to find the projections that capture most of the data variation -after the process the dataset will have only the attributes which capture most of the data variation.

A local way to reduce the dimensionality is to reduce the redundand attributes and eliminate the irrelevant one.

## Feature Subset Selection

To select the attributes to group is possible to approach in many ways:
- Brute force: try all possible subsets as input to data mining algorithm and measure the effectiveness of the algorithm on the reduced dataset
- Embedded Approach: feature selection occurs naturally as part of the data mining algorithm 
- Filter Approach
- Wrapper Approach: a data mining algorithm can choose the best set of attributes

![](/Theory/Images/FeatureSubsetSelection.png)

## Feature Creation
In order to capture more efficiently data charateristics is possible to create new feature with:
- Feature Extraction
- Mapping to a new space
- New Features

# Data Type Conversion
The data type conversion is necessary because most of the time we need numeric features, this can be seen in **Classification** where is necessary to *discretize* nominal values. 
Also makes it easier to discover association rules and requires boolean features.

## Nominal to numeric
A possible way to turn nominal values into numeric values is to use the **One-hot-encoding**.

The OHE implemented by the `sklearn.preprocessing.OneHotEncord` 

## Ordinal to numeric
where the ordered sequence is transformed into consecutive integers, it use `sklearn.preprocessing.OrdinalEncoding`.

## Numeric to binary using the threshold
If the value becomes greater than a threhold it becomes 1 otherwise 0, this is implementes in `sklearn.preprocessing.Binarizer`.

## The Discretization
Some algorithms works better with categorical data, a small number of distinct values can let patterns emerge more clearly and also a small number of distinct values let the algorithm to be less influeced by noise and random effects.

The discretization can be applied with a threshold which can be multiple or binary. 
It is implemented in `sklearn.preprocessing.KBinsDiscretization` with several strategies, such as:
- Uniform
- Quantile
- Means

# Similarities and Dissimilarities

The **Similarity** measures how alike two data objects are, it is a higher value when the object are more alike. 
The **Dissimilarity** is the opposite of the Similarity, both this values falls into the range [0,1]

To measure how similar tow data are is possible to calculate the distance between them using various methods.

## Euclidean Distance

Where D is the number of dimensions and p and q are the attributes on the D-th components the distance is:

$ dist = \sqrt{\sum_{d-1}^{D}(p_d-q_d)^2} $

## Minkowski Distance

D, p and q still represent the same variables as the Euclidean distance, an additional parameter *r* is choosen depending on the dataset.

$ dist = \sum_{d-1}^{D}(|p_d-q_d|^r)^\frac{1}{r}$

The Minkowski distance calculate the absolute value in distance, using as a parameter the **Manhattan Block**.

## Mahalanobis Distance

When considering data distribution the mahalanobis distance between two points p and q decreases if keeping the same euclidean distance the segment connecting the points is stretched along a direction of greater variance of data.

# Data Transformation 

The data transformation is necessary because the features may have different scales and this can alterate the learning techniques also, some learning algorithms are sensitive to feature scaling while others are virtually invariant to it.

Machine learning use *gradient descent* as an optimization technique that require data to be scaled.

## Range-based scaling and standardization
It operate on a single feature and it stretches and shrinks and translate the range according to the range of the feature.

Standardization subtracts the mean and divides by the standard deviation.

in `Scikit-Learn` the transformation are similar and the components used are the `Scaler`.

After the scaling is necessary to **Normalize** the values.

# Imbalanced Data in classification

The performance minority class has little impanct on the standard performance measures. Is possible to estimate the the weigth for each classes using performance measures which allows to take into account the contribution of minority class.

## Cost Sensitive Learning

Several classifiers have the parameter `class_weigth`, it changes the **cost function** and takes into account the imbalancing of the class.
