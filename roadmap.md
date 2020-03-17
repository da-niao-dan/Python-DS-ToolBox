# Data Science for Business
A vast majority of the knowledge is from this [book](https://www.amazon.com/Data-Science-Business-Data-Analytic-Thinking-ebook/dp/B00E6EQ3Xs).

## Main structure of data science at workplace
The Cross Industry Standard Process for Data Mining
See [wiki](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining).

Center of data mining: automated pattern, knowledge and regularities discovery.

### Classic Tasks

* Classification
* Regression
* Similarity matching
* Clustering
* Co-occurrence grouping
* Profiling
* Link Prediction
* Data Reduction
* Causal modeling


### Phases of CISP-DM

* Business Understanding: Formulate the business problem to unambiguous data mining problems
    * What exactly do we want to do?
    * How exactly would we do it?
    * What parts of this use scenario constitute possible data mining models?
* Data Understanding
    * How reliable is the data for our task?
    * What is the cost of getting data?
    * How the data affects our approach? Note that superficially similar tasks could have distinct approaches due to different data available.
    * Business understanding + data understanding determines possible solutions.
* Data Preparation
    * creative, sensible and business minded varialble crafting
    * systematic data processing/clearning
    * Pay Special Attention to Data Leakage
* Modeling
    * Most technical and scietific part. Others are arts. (joking).
* Evaluation: in business context, not in the lab.
    * Quantitative and qualitative assessments
    * Stakeholders considerations: pros and cons
    * Comprehensibility of model, or how to making the model more comprehensible?
    * Do this *Before the deployment*
    * How susceptible is the model to the changing behaviour of data source?
    * The model is what developers build (advisable to include them in data science projects)


### Side Remark: Managing a data science team
* Data science tasks are exploratory undertaking in nature and is closer to research and development than it is to engineering.
* Iterates on approaches and strategy rather than software designs
* Outcomes are far less certain
* Results of each step change change the understandings of problems
* Do not engineeting solution directly for deployment: most of the efforts should go to analytical testings, pilot studies and thowaway prototypes to reduce risks.
* In building a data science team, the most important qualities are:
   * Formulate problems well 
   * Making reasonable assumptions if face of ill-structured problems
   * Prototype solutions quickly
   * Design Experiments that represent good investments
   * Ability to analyze the results
   * NOT traditional software engineering expertise

### Related Skills
* Statistics
* Querying Database
* Data Warehousing
* Machine Learning or Applied Statistics or Pattern Recognition
* Answer Business Questions with These Techniques
    * Who are the most profitable customers? Querying DB
    * Is there really a difference between the profitable customers and the average customer? Hypothesis Testing
    * However, who really are these customers? Can I characterize them? Find pattern that differentiate profitable customers from unprofitable ones.
    * Will some particular new customer be profitable? How much revenue should I expect this customer to generate?

### Summary
* There are fields of study closely related to data science, each task type serves different purpose and has an associated set of solution techniques
* Data Scientist combine these components
* A successful data project involves an intelligent compromise between what the data can do and project goals
* Need to keep in mind how data mining results will be used and use this to inform the data mining process itself.


