# Lab 1: Intro

In Scikit-Learn is possible to represent data as a table.

1 and 2: The iris.csv file is a csv and the separator between the records is a ",". It contains information about the individuals as a form of dataset.

3: No, for the .csv file there is no header. 

```
# Setting the URL for the Practice folder
url = "/home/fabioc/Documents/DataMiningStudyRepo/Practice/"
# Setting the DataFrame
df = pd.read_csv(filepath_or_buffer=url+"iris.csv",names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]) 
```

In this lines of code we simply create a URL path to the folder where all the informations are and then import the `.csv` file and rename all the items columns.

To obtain the column's name we use
~~~
print(df.columns)

## To format the output and tidy it a little:
dims = df.shape
print("The dataset contains "+str(dims[0])+" individuals, with "+str(dims[1])+" attributes each")
~~~
The method `describe` is useful when we need a general description of the dataset:
```
df.describe()

<bound method NDFrame.describe of      sepal_length  sepal_width  petal_length  petal_width           class
0             5.1          3.5           1.4          0.2     Iris-setosa
1             4.9          3.0           1.4          0.2     Iris-setosa
2             4.7          3.2           1.3          0.2     Iris-setosa
3             4.6          3.1           1.5          0.2     Iris-setosa
4             5.0          3.6           1.4          0.2     Iris-setosa
..            ...          ...           ...          ...             ...
145           6.7          3.0           5.2          2.3  Iris-virginica
146           6.3          2.5           5.0          1.9  Iris-virginica
147           6.5          3.0           5.2          2.0  Iris-virginica
148           6.2          3.4           5.4          2.3  Iris-virginica
149           5.9          3.0           5.1          1.8  Iris-virginica

[150 rows x 5 columns]>

This will be the last file of the practice section, because i realized is not necessary
