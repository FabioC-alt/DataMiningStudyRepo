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

**Let's make an example**: \
If we have a market and we want to know the exipiring date, the price and the produce we can aggregate the three charateristics.Thus creating a cube, each cube is an aggregation of the data:

![Alt text](/Theory/Images/AggregationCube.png)

## OLTP and OLAP 
The main differences between **OLTP**(Online Transactional Processing) and **OLAP** (Online Analytics Processing) are mainly related to the goal of the program.

### OLTP
It is used to perform small transaction on the db and each transaction perform a read or write on few records

### OLAP 
It is an interactive data processing system for dynamic multidimentinal analysis, in particular each query involves huge amount of records to process a set of numeric data sum up.
The amount of work change overtime, this is the main analysis model.

## Database VS Data Warehouses
The databases are used by the companies to perform application-based operations, such as reading information about the clients or performing the creations of new records. 
The data warehouses instead are used to perform analysis on the data that are already in the database.

## Data Mart
A data mart is a subset of data stored to a primary data warehouse. Each subset is relevant to a specific business area. 

# OLAP
We will mainly focus our attention on the OLAP system becouse they are the one that are mainly used in the analysis of the data. The use of the OLAP analysis is used to surf the content of a warehouse or data mart. Each session is based on some basic operations, such as:

- Roll up
- Drill down
- Slide and dice
- Pivot
- Drill across
- Drill Through

###  Roll up 
The main idea is to increase the data aggregation
![Alt text](/Theory/Images/RollUp.png)

### Drill Down
In this case the idea is to reduce the data aggregations and have a new level of detail.
![Alt text](/Theory/Images/DrillDown.png)

### Slide and dice
Here we reduce the number of dimension of the information cube.

![Alt text](/Theory/Images/Slice%20and%20dice.png)

### Pivot
A **Pivot** change the layout.
![Alt text](/Theory/Images/Pivot.png)

### Drill Across
Allows to create a link between concepts in interrelated cubes in order to compare them.
![Alt text](/Theory/Images/DrillAcross.png)

### Drill Though
Switches from a multidimensional aggregate data to operational data in sources

# ETL
Extraction Tranformation and Loading is a process that extract, integrate and cleans the data from an operational source to feed the Data Warehouse layer.

Given the fact that we are extracting from different sources we may have some inconsistencies, duplication and/or dirt data, so we need to clean them.
To do so we can perform **Dictionary-based Techniques** which implies having a lookup table to see the corrispondence.
![Alt text](/Theory/Images/DictionaryTech.png)

Also is possible to perform Joins on common attribute to restore the data integrity or the similarity approach (if they looks similar, they may be the same).

After the extraction and cleaning phase we need to perform the tranformation phase, where we perform some opertation in order to adjust the format and reconcile the schema.
This changes can be the date conversion, string conversion, naming, calculation, separations or concatenations of data.

## Denormalization
Implies changing the tables in order to avoid the redundancy:
![Alt text](/Theory/Images/Denormalization.png)

When the data are ready then are uploaded to the data warehouse and this may be done in two different ways:
- Refresh: where we rewrite the entire db 
- Update: only the changes that were made are applied to the data warehouse, all the others remain the same.

# Data Warehouses Architecture

## Main Requirements

The main requirements for a warehouse architecture are:
- Separation between the transactional processing and the analytical
- Scalability, in order to support an increasing number of users 
- Extensibile in the mean that should be capable of manage new application
- Security for monitoring the strategic data that are stored
- Administrability 

## Type of Architecture

### Single-layer Architecture
Only one source layer, the Data warehouse is represented by the middleware abstraction of the db.

![Alt text](/Theory/Images/SingleLayer.png)

## Two-layer Architecture
More than one source layer, usually internal and external. In this case in particular, the data are stored in data mart, which is the main difference with the single-layer architecture.

![Alt text](/Theory/Images/DoubleLayer.png)


## Three-layer Architecture
Here we have the reconciled layer, which is in charge of the materialization of the data obtained using the external data.

![Alt text](/Theory/Images/ThreeLayer.png)

# Conceptual Modeling
The Dimensional Fact Model is a conceptual model created to support the desing and project of the data mart. The main goal of this model are:
- Support the conceptual design
- Create an environment where the user can interact in an easy way with the DWH
- Favor the communication between the users and the designers
- Build a common platform for the logic design
- Give a clear and effective relation of the design of the DWH

The DFM has the following concepts at the base:
- Facts: Relevant concepts
- Measure: Numerical properties related to the facts 
- Dimensions: It is a property of a fact in a predetermined realm to describe a coordination point for the analysis.
- Dimensional Attributes: Dimensions and other possible attributes
- Hierarchy: The relation between father-son related attributes

The facts can be splitted into **Primary Events** and **Secondary Events**.
The primary events are particular occorrence of facts where each of the dimensions have a values assigned, are summaries of primary events wich sum up the result of more primary events.

### Additivity
A measure is called *additive* along with a dimension when you can use the SUM operator to aggregate its values along the dimension hierarchy.
The additivity is the firt charateristic to execute the **Aggregation Operator Classification**.
- Distributive: where we can calculate the global aggregation from the partial
- Algebraic: requiring the usage of additional information in the form of a finite
number of support measures to correctly calculate aggregates from partial
aggregates
- Holistic: calculating aggregates from partial aggregates only via an infinite number
of support measures


➡️ [NextPage](/Theory/Data%20Lake.md)




