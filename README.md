# Python-DS-ToolBox
Python Data Science Toolbox
## What is this note book about?
This is my personal open notebook for Data Science in Python. Its target is people who have no or little experience in Data Science.
Unlike many books, this notebook is quite short and superficial. It covers major steps of doing simple data analysis.
These are simple examples for performing the tasks. Always consult documents, or google when using the functions if you have any doubt.
I learnt the those tools from many sources, welcome to browse and edit this notebook.
This notebook is just a **remainder for what is available out there**. It does not offer any mathematical or technical or business details. 
You can visit this [handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) for technical details. 

## Quick advice
There are many great sources to learn Data Science, and here are some advice to dive into this field quickly:

0. Get some basic foundation in statistics and python. See www.hackerrank.com.
1. Get to know the general idea and areas of knowledge about data science.
2. Practice as you go. Google or Bing any term or question you don't understand. Stackoverflow and supporting documents are your best friend. During this process, do not lose sight of the big picture of data science.
3. Become a better data scientiest by doing more projects! (Don't try to memorize these tools, just do data science!)
4. Once comfortable with all materials in this notebook and engage in data analysis in business, you will know what skills to pick up next.

## Materials in this notebook

0. [Environment Configuration](#Environment-Configuration)
1. [Data Processing](#Data-Processing)
    0. Getting Data
        0. Loading Files
        1. From APIs
        2. From SQL database
    1. [Organizing Data](#Organizing-Data)
        0. Data Cleaning
            0. Processing Datetimes
            1. Processing Lists
        1. Transform DataFrames
2. Exploring Data
    0. Simple Data Visualization
    1. Simple Statistical Tools
3. Communicating with Data
    0. Interative Data Visualization
4. Data_Models
    0. Linear_Models
5. Other important skills
    0. Linux and Bash shells
    1. Efficient Coding in Python
    2. Data Structure and Algorithms
    3. Parallel and Cloud Computing with Spark technologies

---

# <a name="Environment-Configuration"></a> Environment Configuration
Choice of Working Environments: I recommend *VScode* and *WSL* for projects, *Jupyter Notebook* for prototyping.
Manage of python packages: I recommend *Anaconda* for this task.
Manage of version control and collaboration: Use *git* for personal projects, Use *Github* for open projects, *GitLab* for industrial projects.

Google or bing these keywords to find revelvant tutorials. This will be the last reminder of such actions.

# <a name="Data-Processing"></a> Data Processing
## Getting Data
### Read Data From Files
```
import pandas as pd
```
#### csv file
csv stands for comma-separated values. Our target is to import csv file into a PandasDataFrame.
The first step is to understands what is the delimiter of the file.
Then we use the function read_csv in pandas.

Suppose f is a csv file with tab as delimiters, then we do this:
```
df = pd.read_csv(f,delimiter ='\t')
```

#### json files
json files are human readable, you can think of them as python dictionaries. We want to load them into pandas DataFrame.
The furst step is to view your data to get a sense.
The second step is to use a function called json_normalize.
```
from pandas.io.json import json_normalize

df = json_normalize(yourjsonfile,['key1', 'key2', ...], max_level=3)
```

### Get data from APIs
Before we begin, have some basic knowledge on how the internet works and what is http, and what is API.
Getting data from web API usually require reading API manuals for specific APIs.
I recommend using package requests for this task in Python.

```
# Assign URL to variable: url

url = 'yoururl' # Consult your api documents here.

header = {"Accept-Encoding":"gzip"} # Consult your api documents and requests package website here.

# Package the request, send the request and catch the response: r
param_dict = {your_parameters}# Consult your api documents and requests package website here.

with requests.Session() as s: 
    download = s.post(url, data=param_dict,headers=header) ## consult post, get, and other http requests.
```    
### Get data from a SQL database
Here I will use a MySQL database for example. We are going to get data from MySQL database using SQL commands in python.

There are abundant choices of packages here. I use mysql-connector package.
```
import mysql.connector
import pandas as pd


cnx = mysql.connector.connect(user='yourusername', password='yourpassword',
                              host='yourhostaddress',
                              database='yourdatabase') ## some arguments are optional!
cursor = cnx.cursor()
query = ("""
YourSQLCode Here
""")

cursor.execute(query)
table_rows = cursor.fetchall()
df= pd.DataFrame(table_rows, columns=cursor.column_names)
cnx.close()
```

## Organizing Data
### Data Cleaning
Basic str manipulations are quite handy in data cleaning.
#### Processing Datetimes

```
## This document displays common tools for dealing with datetime data

from datetime import date
from datetime import datetime

### Example1
## Suppose we have a list of strings, representing datetimes, we are going to convert them into datetime objects
example1 = ['2019-12-09 00:00:00', '2019-12-10 00:00:00', '2019-12-11 00:00:00', '2019-12-12 00:00:00', '2019-12-13 00:00:00']

## First we identify time formats
time_format = '%Y-%m-%d %H:%M:%S'

## Then we use strptime method of datetime class to convert them into datetime objects
example1_datetime_list = list(map(lambda x:datetime.strptime(x,time_format),example1))

## We can check out date using ctimp method
## Let's check out the first date in our list
example1_datetime_list[0].ctime()

## Sometimes we only concern about date.
## The classic format called 'iso format' for date is 'YYYY-MM-DD'
## Let's check out the first date in our list in iso format:
example1_datetime_list[0].date().isoformat()
```

#### Processing Lists
```
### This document introduces some common operations of lists in Python

## Concatenate list of lists:
## Suppose x is an example list
x=[[1],[2],[3]]
## If we want [1,2,3] we can do:
new_list=[x[i][j] for i in range(len(x)) for j in range(len(x[i]))]
```


### Transform DataFrames
```
import pandas as pd

## Suppose df is a loaded DataFrame, 'column_name' is the name of one of its columns.

## Set a column as index column

df.set_index('column_name')


## suppose you have df1 and df2 and they both have a column named 'column_name'
## You can merge them using inner join. Save the merged DataFrame as df.
df = df1.merge(df2,how='inner',on='column_name')


```
