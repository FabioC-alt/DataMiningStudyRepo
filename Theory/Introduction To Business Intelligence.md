# Introduction To Business Intelligence

The definition of business intelligence is an umbrella term that includes the applications, infrastructures and tools, and the best practice that enable access to and analysis of information to improve and optimize decisions and performance.

The business intelligence main focuses on the transformation of the raw data into useful information. It aggregate, cleans and transform them. 

To support the business intelligence process an ad hoc infrastructure is necessary. The main passage that needs to be supported into the business intelligence is the *Data Staging*. During this step the data are copied into a warehouse system in order to avoid any interfering with the original data. In the warehouse it is possible to perform many important operations:

- OLAP Analysis
- Reporting
- Data Mining 
- Reconciliation

## The Data WareHouse

The Data Warehouse is the main instrument that supports the operation of business intelligence. This components is a repo of data optimized to for the decision making process and it has many advantages:

- It is possible to manage the history of the data
- There is the possibility to make multidimensional analysis
- Based on a simple model that can be easely undestand by the new users of the system

A data warehouse has the following characteristics:

- It is **subject oriented**, this means that it is focused on the interests of the client that is using it.
- It is **integrated and consistent**: this means that integrates data from different sources and uniform the data that are inserted
- It allows to have a **vision of the evolution** of the data over time

## Multidimensional Model
The multidimensional model allows users to interactively navigate the data warehouse information exploting the multidimensional model. Typically the data are analyzed at different levels of aggregation.

**Let's make an example**: If we have a market and we want to know the exipiring date, the price and the produce we can aggregate the three charateristics.Thus creating a cube, each cube is an aggregation of the data:

![Alt text](/Theory/Images/AggregationCube.png)

