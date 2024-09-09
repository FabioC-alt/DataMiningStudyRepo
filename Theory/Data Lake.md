# Data Lake

The data once stored in the data warehouses are still not usable for the decision making process, this data are called **Dark Data**. Most of them will never be used, it is estimed that most of them are useless and another part is redundant, thougth only a small part of these data is *Business-critial*.

## What is a Data Lake?
A repository of data stored in its raw format, still not cleaned. So a data lake is a structure where all the files that do not have a precise use are accumulated in order to wait for processing.

# Traditional Data Analytics Architecture vs. Modern Data Lakes

## 1. Siloed Departmental Data
- **Traditional Approach**: Data is stored in isolated systems specific to departments (e.g., sales, finance, marketing). This leads to fragmented data that is difficult to consolidate and analyze across the organization.
- **Problem**: Limits cross-functional data insights and creates bottlenecks for organization-wide analytics.
- **Modern Approach**: Data lakes store all data from different departments in a centralized repository, making it easier to break down silos and perform organization-wide analysis.

## 2. Enterprise Data Warehouse (EDW) with Long Implementation Cycles
- **Traditional Approach**: Implementing an Enterprise Data Warehouse (EDW) requires significant planning, integration of data sources, ETL (Extract, Transform, Load) processes, and upfront schema definition.
- **Problem**: Long and expensive implementation cycles; difficulty adapting to changing business needs.
- **Modern Approach**: Data lakes allow quick ingestion of raw data without extensive preprocessing. Schema and transformations are applied when needed (schema-on-read), resulting in faster implementation and adaptability to changing requirements.

## 3. Limits of Scalability and High Costs
- **Traditional Approach**: Scaling traditional databases and data warehouses can be expensive, as both storage and compute capacity must be upgraded together.
- **Problem**: High cost and lack of flexibility when scaling infrastructure.
- **Modern Approach**: Data lakes (especially cloud-based) provide highly scalable storage at a low cost, with the ability to scale compute resources independently of storage.

## 4. Compute and Storage Have to be Scaled Together
- **Traditional Approach**: In a traditional EDW, compute and storage are tightly coupled, meaning you need to increase both even if only one is required.
- **Problem**: Lack of flexibility and cost-efficiency.
- **Modern Approach**: Data lakes decouple storage from compute, allowing organizations to scale them separately based on specific needs.

## 5. Lack of Support for Unstructured Data
- **Traditional Approach**: Traditional databases are optimized for structured data (e.g., tables and columns), making it difficult to handle unstructured or semi-structured data (e.g., images, logs, videos).
- **Problem**: Limited ability to work with diverse data types.
- **Modern Approach**: Data lakes handle all types of data—structured, semi-structured, and unstructured—providing greater flexibility and better data utilization.

## 6. Machine Learning (ML) and Data Science (DS) Workloads
- **Traditional Approach**: Traditional data analytics platforms are not optimized for machine learning and data science workloads, which require access to large amounts of raw data.
- **Problem**: Limited support for modern data science tools, requiring specialized environments that are difficult to integrate.
- **Modern Approach**: Data lakes are better suited for ML/DS workloads, storing raw data for feature engineering, model training, and experimentation.

## 7. Ad-hoc Analytical Queries
- **Traditional Approach**: Data warehouses are often optimized for predefined, structured queries, making ad-hoc queries slow or inefficient.
- **Problem**: Difficulty running fast, ad-hoc queries common in exploratory analysis and data discovery.
- **Modern Approach**: Data lakes provide access to raw data for analysis, allowing users to run ad-hoc queries on large datasets using modern analytics engines (e.g., Apache Spark, Presto, Athena).

## Requiremnts
The main requirements for this approach are:
- Wide range of analytics use cases
- Data in hands of the business users
- Flexible and scalable data architecture
- Short Time to value
- Holistic metadata managment, governance, security monitoring and usage analytics

# Features and Use Cases of Data Technologies

| **Use Cases**                                      | **Tech Solutions**      | **Feature Trends**                                         |
|----------------------------------------------------|-------------------------|------------------------------------------------------------|
| **Mission-critical, low latency, insight apps**    | Data Warehouse / Hot     | - More expensive HW/SW<br> - Use case-specific data<br> - Low latency<br> - More governance<br> - Higher data quality<br> - Used by end-users and data analysts |
| **Agile insight apps**                             | Data Hub / Warm          | - Balances cost and performance<br> - Agile insight generation for quick decision making |
| **Staging area, data mining, searching, profiling, cataloging** | Data Lake / Cold         | - Less expensive HW/SW<br> - All enterprise data<br> - Higher latency<br> - Less governance<br> - Lower data quality<br> - Used by data scientists |

# Detailed Requirements: Traditional Data Systems vs Insight-Driven Systems

| **Aspect**                        | **Traditional Data Systems**                                      | **Insight-Driven Systems**                                                |
|-----------------------------------|-------------------------------------------------------------------|----------------------------------------------------------------------------|
| **Data Sources**                  | - Structured, relational data from transaction systems<br> - Relational and operational data stores | - Traditional sources<br> - Semi and unstructured sources: logs, websites, social media, alternative data providers |
| **Data Movement (Ingestion)**     | - Amount of data that could be moved is limited                   | - Virtually unlimited amount of data that could be moved into the system in original form at a required latency |
| **Storage**                       | - Limited volume of stored data                                   | - Unlimited volume of data. Source of truth for all source data              |
| **Data Structure**                | - Schema is designed upfront, before data is captured<br> - Data format dictated by storage/technology | - Schema is not fixed when data is captured<br> - Variety of supported open formats for analytics pipelines (relational, document, graph, etc.) |
| **Data Transformations**          | - Upfront, time-consuming data cleansing, enrichment, integration, and transformation<br> - “Single source of truth” of trusted and widely usable data | - Add transformation for ad-hoc data querying and data science related feature engineering |
| **Analytics**                    | - SQL Queries<br> - BI tools<br> - Full text search              | - Self-service BI<br> - Big data analytics<br> - Real-time analytics<br> - Machine learning<br> - Data exploration/visualization<br> - Allow users to securely explore and query raw data<br> - Easily introduce new types of analytics |
| **Price/Performance**            | - Highest cost storage<br> - Fastest query results                | - Low-cost storage<br> - Performance scale/speed/cost tradeoffs             |
| **Users**                        | - Business (analysts)                                            | - Data Scientists<br> - Data analysts and engineers<br> - Developers         |
| **Data Quality**                 | - High                                                            | - Low and high, depending on use case<br> - Data quality must be transparent |
| **Data Sharing and Collaboration**| - Very limited<br> - Mainly via central DW/BI team                | - Rich sharing<br> - Raw and transformed data sets<br> - Analytical models<br> - Dashboards can be easily and securely shared |


## Type of users in Data Lake

- **Business Users**: use a pre-configured dashboard and report
- **Business Analyst**:use a self-service BI and ad-hoc analytics, built own models to provide business insigth
- **Data Scientist, Engineering, App Dev**: perform statistical analysis and ML training, implement big data analytics to identify trends, solve business problems, optimize performance.

