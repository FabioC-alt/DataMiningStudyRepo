# Cheat Sheet for memorizing the exam

## Imports and Files

Imports:
```
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import tree #Classification
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