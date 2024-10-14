# Association Rule

Given a set of transaction it is possible to find specific rules that will predict the occurence of an item based on the occurrance of another items. 

A possible example is the association: {Diaper, Milk} -> {Beer}. 
If the Diaper and Milk are more selled, it is expected an increase in Beers sold. If a rule is low supported it means that it generates a random association and is note reliable. Instead if the rule has low support but high confidence it can represent an uncommon but interesting phenomenon.

There are many ways to investigate the association rules:

## Brute Force Approach

List all the possible association rule and compute for each rule the confidence level. This approach can result in a prohibitive computational analysis caused by the big amount of data that needs to be analized.

To speed up this approach is possible to prune the fail branches.

When analyzing an association rule it is possible to follow a two step approach:

1. Frequent Itemset Generation
2. Rule Generation

The first step is the most computationally expensive:
Each itemset in the lattice is a candidate frequent itemse and it is possible to calculate the **support** and the **confidence** for each of these itemset. The computational complexity is given by the fact that the big O is : $O(NWM)$ where N is the number of transaction globally, W is the mean number of element per transaction and M is the number of candidates.
To reduce the complexity of this calculation is possible to cut the tree by reducing the number of candidates:

## Apriori Algorithm
To reduce the number of candidates is possible to use the **Apriori** algorithm, it exploits the fact that the *sup* is antimonotone, which means that the itemset which is frequent in all the subset is also frequent.

If a subset is less frequent, it means that all the itmset which include that item are less frequent, so they can be eliminated.

## Rule Generation
Once all the itemset are found it is necessary to find the rule that associate each candidate of the itemset to the other.

To reduce the analysis of the rules is possible to prune the based on the confidence of each rule. 

## Mono-dimensional vs Multi-dimensional rule

The mono-dimensional rule associate the candidate with each other, (eg. each transaction), while the multi-dimensional associate the tubles

## Multilevel Association Rules
In a real analysis the rules are associate using meta-levels in order to predict an entire set of items based on another, and it is also possible to use a *specialized* or a *generalized* approach.