# Tidy Data

**Tidy Data** is a framework useful to create an organized and ordered dataset.
In tidy data each variable is a column and each observation is a row, while each type of observational unit forms a table. An important factor is the use of the order of the observation compared to the order of variables. 

In the order the fixed variable, also called **dimensions**, should be posed first, while the measured variables should comes later. 

# Tidying messy dataset

The most common problems with messy dataset are these 5, along with the remedies:

- Column headers are values, not variable names
- Multiple variables are stored into one column
- Variables are stored in both rows and column
- Multiple types of observational units are stored in the same table
- A single observational unit is stored in multiples tables

## Column headers are values, not variable names

Here the variables are three: religion, income and frequency. The variables should all be column. So the cloumn should be turned into rows. 

![](/Practice/Images/ReligionAndIncome.png)

The religion, the income and the frequency are called **colvar**, from the join of the two words *column* and *variable*. 
To tidy this dataset, the procedure is called *melt*, once molted the dataset will look like different tables for each religion.

![](/Practice/Images/TidyReligionAndIncome.png)

Another example is this one:

![](/Practice/Images/Music.png)

Here the variables are: Song, year, artist, track, time, data.entered. The observations are the weeks.

So the weeks should be turned into rows.
Obviously now the dataset is less readable, but it is more suitable for data analysis.

![](/Practice/Images/TidyMusic.png)

## Multiple variable stored into one column
Another situation that may be present is the one where the variables are mixed in the same column, like this one, where the sex and the age are stored in the same variable:

![](/Practice/Images/AgeAndSex.png)

The two step to perform are the melting, which leads to a more organic way to look a the dataset, and then tidy it by splitting the Age-Sex column into two different column.

![](/Practice/Images/TidyAgeAndSex.png)

This way is possible to rearrange the data in oder to be suitable for the data analysis, anyway the tidy one is better because the age

## Variables are stored both in columns and rows

In this case the variable are stored both in columns and rows, particular the 

based on the Harley Whicknam paper.