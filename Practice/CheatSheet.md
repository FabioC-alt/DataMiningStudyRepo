# Data Elaboration and Preparation

## Imports and Files

Imports:
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

Varibles: 
``` 
file_name = 'File_name.csv'
separator = 'separator'
random_state = 42
target = 'Class_target'
```

Directives
```
%matplotlib inline
np.random.seed(rando_state)
```


File Existence Control
```
if not fileVar.is_file():
    from google.colab import files
    print('Select input file for train and test')
```

Read CSV files:
```
# Load file (Prima riga ci sono le label e la prima colonna ha gli indici)
df = pd.read_csv(file_name, delimiter = separator, header = 0, index_col = 0)

# Load file (DataSet senza label e indici)
df = pd.read_csv(file_name, delimiter = separator, header=None, index_col=None)

# Load file (DataSet con names)
df = pd.read_csv(file_name, delimiter = separator, header=None, index_col=None, names=['colonna1', 'colonna2'])

# Load file (with index but without column name)
col_names=['Index', 'Sex', 'Length', 'Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
df=pd.read_csv(file_name, sep=separator, header=None, names=col_names, index_col=['Index'])

# Load data from a .txt file
text = np.loadtxt(file_name, delimiter = separator)
df = pd.DataFrame(text)
```
## Name Assigning Column

```
# assegnare dei nomi alle colonne se in dataset originale non ha nomi alle colonne
columns =[]
for i in range(df.shape[1]):
    columns.append(str(i)) # ['0','1' .... ]

df.columns = columns

# assegnare dei nomi alle colonne se in dataset originale non ha nomi alle colonne
columns =[]
for i in range(df.shape[1]):
    columns.append(str(i)) # ['0','1' .... ]

# last element
columns[-1] = 'Class_target'
df.columns = columns

```

## Data Description
```
# Show the DataFrame (All)
df

# Show Structure
df.describe()

# Show the head of the dataframe
df.head()

# For each column show the frequencies of each distinct value
np.unique(df, return_counts = True)

# Show the number of rows and columns
print(f"There are {df.shape[0]} rows and {df.shape[1]} columns in this dataset")

# Show Shape
print ("The shape is: {}".format(df.shape))

# Show the size of the dataframe
print(f"The dataframe has size: {df.size}")

# Pairplot (relazioni fra attributi rispetto al target)
# NON TIENE VALORI STRINGHE (NO ERRORI)
sns.pairplot(df, hue = target)

# Boxplot (trovare Outliers)
# NON TIENE VALORI STRINGHE (DA ERRORI, DA TOGLIERE)
plt.figure(figsize=(15,15))
pos = 1
for i in df.columns:
        if(type(df[i][0]) != str):
                plt.subplot(4, 3, pos)
                sns.boxplot(df[i])
                pos += 1

# Boxplot
# Drop column stringa
df_for_boxplot = df.drop(['Column_containing_string_type'], axis=1)
plt.figure(figsize=(15,15))
pos = 1
for i in df_for_boxplot.columns:
    plt.subplot(3, 4, pos)
    sns.boxplot(data=df[i])
    pos += 1

# Correlation Matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, cmap="YlGnBu", annot=True)

#Check the number of rows with missing values
rows_missingvalues = df.isna().any(axis=1).sum()
print("Rows with missing values: {}".format(rows_missingvalues))

# Histogram of numeric data
pd.DataFrame.hist(df, figsize=[15,15]);

# Histogram of the column target (even if a string)
df['target'].hist()

# Scatter Plot (X column 0 and Y column 1 of df)
sns.scatterplot(x=focus[0], y=focus[1], data=df, hue="target")
```
## Dataset Modification
```
# Merge the two dataframes with the 'outer' how, as to perform a SQL-like full outer join
# on the two indexes, adding suffixes as requested (default option)
# (Entrambi hanno Indici e prima riga Label da differenziare Target)
df = first_df.merge(second_df, how = 'outer', left_index = True, right_index = True, suffixes = ('_x', '_y'))

# Drop those rows from the dataframe
df = df.drop(index = indexes_to_delete, axis = 0)

# Drop specific column
df = df.drop(columns = 'Column_Name', axis = 1)

# Drop more than 1 column
df = df.drop(columns = ['Column_Name1', 'Column_Name2'], axis = 1)

# Rename specific column
df = df.rename(columns = {'Old_Name1':'New_Name1', 'Old_Name2':'New_name2'})

# Get the column names
column_names = list(df.columns)

# Reindex the dataframe
df = df.reindex(columns = column_names)

# Eliminate the rows containing null values
df = df.dropna()

# Delete row where value in column 1 is different from column 2
df = df.drop(df[df['class_x'] !=  df['class_y']].index)
```
## Example of Comments
```
# We can see that there are some distributions that are very similar and higly correlated (such as Length/Diameter
# or the different weights) and there is also a significant presence of outliers.
# All the weight attributes are skewed on the left and have a long tail.
# Also, our data contains some missing values.
# All this things can compromise our analysis so it's time to pre-process.
```

## Data Transformation
```
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

column_target = 'target'

# Set the transformer data type (if required)
transf_dtype = np.int32

# machineLearning-03c-prepr-dissim.pdf 25/71
# Specific columns dataframe to one hot encoding , add new columns drop old column
# OneHotEncoder (da Nominal a Numerical)
# from 1 column to n column of 0/1
encoder = OneHotEncoder(dtype = transf_dtype)
transformed = encoder.fit_transform(df[[column_target]])
df[encoder.categories_[0]] = transformed.toarray()
df = df.drop(column_target, axis = 1)

# Specific column dataframe to one hot encoding , inplace
# OrdinalEncoder (da Ordinal a Numerical)
encoder = OrdinalEncoder(dtype = transf_dtype)
df[column_target] = encoder.fit_transform(df[[column_target]])

# All dataframe to one hot encoding
# from Nominal to Numerical
transf_dtype = np.int32
encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False, dtype = transf_dtype)
# Fit and transform the data
X_e = encoder.fit_transform(df)
X_ohe = pd.DataFrame(X_e)
X = X_ohe

#Transform categorial data(Sex) into new boolean attributes
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
# Set the transformer data type
transf_dtype = np.int32

# Specific columns dataframe to one hot encoding , add new columns drop old column
# Instantiate the encoder only on needed columns and perform `fit_transform`
encoder = make_column_transformer((OneHotEncoder(handle_unknown = 'ignore', sparse = False, dtype = transf_dtype), ['Sex']), remainder='passthrough')
transformed = encoder.fit_transform(X)

#Since `fit_transform` returns an `ndarray`, but a dataframe is needed
# Column Sex has M,F,I value
encX = pd.DataFrame(transformed, columns = encoder.get_feature_names())
encX.rename(columns = {'onehotencoder__x0_F':'Female', 'onehotencoder__x0_I':'Indefinite', 'onehotencoder__x0_M':'Male'}, inplace = True)                   #renaming of the freshly added columns
encX

# machineLearning-03c-prepr-dissim.pdf 26/71
# OrdinalEncoder (from Ordinal to Numerical )
# In order to do a classification, our column_to_convert column has to become numerical
# from 1 column to 1 column of range -1 to +1
encoder = OrdinalEncoder()
df['column_to_convert'] = encoder.fit_transform(df['column_to_convert'].values.reshape(-1,1))
```
# Classification

## X and y Dataset

```
from sklearn.model_selection import train_test_split

X = df.drop(target, axis = 1)
y = df[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,train_size = 2/3, random_state = random_state)
print(f"We have {Xtrain.shape[0]} items in our training set")
print(f"We have {Xtest.shape[0]} items in our test set")
```
## Train Validation Test

```
Xtrain2, Xval, ytrain2, yval = train_test_split(Xtrain, ytrain, random_state= random_state)
print(f"We have {Xtrain2.shape[0]} items in our train set")
print(f"We have {Xval.shape[0]} items in our validation set")
```

```
Xtrain2, Xval, ytrain2, yval = train_test_split(Xtrain, ytrain, random_state= random_state)

print(f"We have {Xtrain2.shape[0]} items in our train set")
print(f"We have {Xval.shape[0]} items in our validation set")
```

# Trees

## Decision Tree
```
# dt comes from a previous creation of a DecisionTree Classifier
default_depth = dt.tree_.max_depth
range_depth = range(1, default_depth+1)
#use accuracy as method of evaluation
scores= []
for i in range_depth:
    current_model = DecisionTreeClassifier(criterion="entropy", max_depth=i, random_state=random_state)

    current_model.fit(Xtrain2,ytrain2)
    yval_predicted = current_model.predict(Xval)
    scores.append([i, accuracy_score(yval, yval_predicted)*100])
print(scores)
```

## Tree Accurancy
```
#now we insert the scores in a dataframe to get the best parameters easily
score_df = pd.DataFrame(data=scores, columns=["max_depth", "accuracy_score"])
#order dataframe to get best accuracy score
score_df = score_df.sort_values(by=["accuracy_score"], ascending=False)
score_df.head(1)
```

## Generic Algorithm
