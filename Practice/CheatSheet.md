# Cheat Sheet for memorizing the exam

## Imports and Files

Imports:
```
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import tree #Classification
import seaborn as sns
```

Files includes
```
fileVar = Path('pathToFile.csv`)
```

File Existence Control
```
if not fileVar.is_file():
    from google.colab import files
    print('Select input file for train and test')
```
Read CSV files:
```
values = pd.read_csv(fileVar)
```

## Training

First is necessary to create the dataset without the class
```
X_train = train.drop(labels=`Class`, axis=1) #Dropping a column using its name or
X_train = train.drop(train.colums[-1], axis=1) #Dropping a column using its index
X_train.shape #Always control the shape of the training dataset, it should be one column less
```

Then the prediction output to evalueta the test

```
#y_train = train.drop(train.columns[0:-1], axis = 1) # drop all the columns but the last
y_train = train.iloc[:,-1] # DEPRECATED, check the code on Virtuale, but with this works unfortunately :(
y_train.shape
```

### Tree training 

```
# Check the imports for the tree class
model = tree.DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)
```
The fit method creates a decision tree built on top of the two subset given.
Then we check the predictions
```
y_predicted_train = model.predict(X_train)
```

And then print the result formatted and based on the mean
```
accuracy_train = np.mean(y_train == y_predicted_train) * 100
print("The accuracy on training set is {0:.1f}%".format(accuracy_train))
```

Now let's control it using the test subset
```
test=pd.read_csv(in_test)
X_test = test.drop(test.columns[-1], axis = 1) # drop the last column
y_test = test.iloc[:,-1] # keep only the last column (DEPRECATED)
y_predicted_test = model.predict(X_test)
accuracy_test = np.mean(y_test == y_predicted_test) * 100
print("The accuracy on test set is {0:.1f}%".format(accuracy_test))
```

## Drawing a Tree
```
from matplotlib.pyplot import figure

figure(figsize=(25,25))
tree.plot_tree(model, rounded=True,filled=True,Class_names=['False','True'])

```

This instructions plot the tree.
The `plot_tree` method takes as input 4 parameters:
- **model**: is the model used to predict the result
- **rounded=True**: rounds the box shape
- **filled=True**: fills the boxes with color
- **Class_names**: gives a name to the classes, given that the prediction are 1s and 0s, they can be labeled as True and False.

# Histograms
```
# To show all the columns 
pd.DataFrame.hist(df
                  , figsize = [10,10]
                 );
    
# To show only the target one
### adjust the line below
plt.hist(df[target])
plt.show()

```

# Pairplot

```
sns.pairplot(df, hue='quality', diag_kind='kde')
```

The `hue` parameter is used to add color grouoping to the plot

The `diag_kind` parameters control the type of plot that appears on the diagonal of the pair plot, and `kde` stands for **Kernel Density Estimate** and an alternative is `hist`, to show the histograms instead of KDE plots.

# Correlation 
```
corr = df[df.columns].corr() # Deposit all the correlation information
plt.figure(figsize=(15,10))  # Figsize
sns.heatmap(corr, cmap='YlGnBu`, annot= true) # Creates the heatmap
```


