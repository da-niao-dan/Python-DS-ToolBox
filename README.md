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
2. Practice as you go. Google or Bing any term or question you don't understand. Stackoverflow and supporting documents for specific packages and functions are your best friend. During this process, do not lose sight of the big picture of data science.
3. Become a better data scientiest by doing more projects! (Don't try to memorize these tools, just do data science!)
4. Once comfortable with all materials in this notebook and engage in data analysis in business, you will know what skills to pick up next.

## Materials in this notebook

0. [Environment Configuration](#Environment-Configuration)
1. [Data Processing](#Data-Processing)

    * Getting Data

        * By Loading Files
        * From APIs
        * From SQL database
    * [Organizing Data](#Organizing-Data)
        * Take a First Look
        * Data Cleaning
            * Processing Datetimes
            * Processing Lists
        * Transform DataFrames

2. [Exploring Data](#Exploring-Data)

    * Simple Data Visualization
    * Simple Statistical Tools

3. [Communicating with Data](#Communicating-with-Data)

    * Interative Data Visualization

4. [Data Models](#Data-Models)

    * Linear_Models

5. Other important skills

    * [Linux and Bash shells](#Linux-and-Bash-shells)
    * Efficient Coding in Python
    * Data Structure and Algorithms
    * Parallel and Cloud Computing with Spark technologies

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

Here, I will also include some techniques about joining tables using SQL code.


Self-join is just treat the same table as two tables. 

```
## Suppose populations is a table with column size, country_code and year
SELECT p1.country_code,
       p1.size AS size2010, 
       p2.size AS size2015,
-- 2. From populations (alias as p1)
FROM populations AS p1
  -- 3. Join to itself (alias as p2)
  INNER JOIN populations AS p2
    -- 4. Match on country code
    ON p1.country_code = p2.country_code
        -- 5. and year (with calculation)
        AND p1.year = p2.year - 5;
```

Case when and then:
Sometimes you want to put numerical data into groups, it's time to use CASE WHEN THEN ELSE END commands.




## Organizing Data
### Take a First Look
Before you do anything to your newly gained data, it's always a good idea to look at it first.

```

import pandas as pd

## Suppose the data is loaded in a Pandas DataFrame data

## Look at the table as a whole:
data.info()
data.head()
data.describe()

```

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

## Convert timedelta to floats for calculations
aFloat = example1_datetime_list[0]/np.timedelta64(1,'D')
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

# <a name="Exploring-Data"></a> Exploring Data
## Simple Data Visualization
There are many tutorials for the famous *matplotlib* package for plotting in Python.
Therefore, here I will present an alternative package for plotting: *bokeh*.

I prefer Bokeh because it supports interactive design.

```
## Suppose data is a pandas DataFrame with our data and our goal is to plot a scatter plot of data.column1 against data.column2

# Perform necessary imports
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.column1,
    'y'       : data.column2
})

# Create the figure: p
p = figure(title='graph', x_axis_label='column1', y_axis_label='column2',
           plot_height=400, plot_width=700)

# Add a circle glyph to the figure p
p.circle(x='x', y='y', source=source)

# Output the file and show the figure
output_file('gapminder.html')
show(p)

```


# <a name="Communicating-with-Data"></a> Communicating with Data
## Interactive Data Visualization

First, we make a plot.
```
### Suppose data is a pandas DataFrame, with certain row and column labels

# Import the necessary modules
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[SomeRowLabel].col1,
    'y'       : data.loc[SomeRowLabel].col2,
    'y2'      : data.loc[SomeRowLabel].col3,
    'y3'  : data.loc[SomeRowLabel].col4,
    'y4'      : data.loc[SomeRowLabel].col5,
})

# Save the minimum and maximum values of the col1 column: xmin, xmax
xmin, xmax = min(data.col1), max(data.col1)

# Save the minimum and maximum values of the col2 column: ymin, ymax
ymin, ymax = min(data.col2), max(data.col2)

# Create the figure: plot
plot = figure(title='Data for SomeRowLabel', plot_height=400, plot_width=700,
              x_range=(xmin, xmax), y_range=(ymin, ymax))

# Add circle glyphs to the plot
plot.circle(x='x', y='y', fill_alpha=0.8, source=source)

# Set the x-axis label
plot.xaxis.axis_label ='col1 (name)'

# Set the y-axis label
plot.yaxis.axis_label = 'col2 (name2)'

# Add the plot to the current document and add a title
curdoc().add_root(plot)
curdoc().title = 'document name'



```

Second, we enhance the plot with some shading.

```
# Make a list of the unique values from the col5 column: col5s_list
col5s_list = data.col5.unique().tolist()

# Import CategoricalColorMapper from bokeh.models and the Spectral6 palette from bokeh.palettes
from bokeh.models import CategoricalColorMapper
from bokeh.palettes import Spectral6

# Make a color mapper: color_mapper
color_mapper = CategoricalColorMapper(factors=col5s_list, palette=Spectral6)

# Add the color mapper to the circle glyph
plot.circle(x='x', y='y', fill_alpha=0.8, source=source,
            color=dict(field='col5', transform=color_mapper), legend='col5')

# Set the legend.location attribute of the plot to 'top_right'
plot.legend.location = 'top_right'

# Add the plot to the current document and add the title
curdoc().add_root(plot)
curdoc().title = 'titleOfDocument'
```

Third, we can add slider to vary the rowlabel (of some attributes). E.g. Year, age, etc..

```
# Import the necessary modules
from bokeh.layouts import widgetbox, row
from bokeh.models import Slider

# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Set the rowvalue name to slider.value and new_data to source.data
    rowvalue = slider.value
    new_data = {
        'x'       : data.loc[rowvalue].col1,
        'y'       : data.loc[rowvalue].col2
        'y2' : data.loc[rowvalue].col3,
        'y3'     : data.loc[rowvalue].col4,
        'y4'  : data.loc[rowvalue].col5
    }
    source.data = new_data
    plot.title.text = ' Plot data for %d' % rowvalue


# Make a slider object: slider
slider = Slider(start=initRowVal,end=endRowVal,step=1,value=initRowVal, title='AttributeLabel')

# Attach the callback to the 'value' property of slider
slider.on_change('value',update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)
```

Fourth, we could add more interactivity.
Try these out to get a sense about what are they doing:

HoverTool:
```
# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool: hover
hover = HoverTool(tooltips=[('Y2', '@y2')])

# Add the HoverTool to the plot
plot.add_tools(hover)
# Create layout: layout
layout = row(widgetbox(slider), plot)

# Add layout to current document
curdoc().add_root(layout)
```
Dropdown Manu:

```
# Define the callback: update_plot
def update_plot(attr, old, new):
    # Read the current value off the slider and 2 dropdowns: rowval, x, y
    rowval = slider.value
    x = x_select.value
    y = y_select.value
    # Label axes of plot
    plot.xaxis.axis_label = x
    plot.yaxis.axis_label = y
    # Set new_data
    new_data = {
        'x'       : data.loc[rowval][x],
        'y'       : data.loc[rowval][y],
        'y2' : data.loc[rowval].col3,
        'y3'     : data.loc[rowval].col4,
        'y4'  : data.loc[rowval].col5
    }
    # Assign new_data to source.data
    source.data = new_data

    # Set the range of all axes
    plot.x_range.start = min(data[x])
    plot.x_range.end = max(data[x])
    plot.y_range.start = min(data[y])
    plot.y_range.end = max(data[y])

    # Add title to plot
    plot.title.text = 'Plot data for %d' % rowval

# Create a dropdown slider widget: slider
slider = Slider(start=beginVal, end=endVal, step=1, value=beginVal, title='AttributeName')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Create a dropdown Select widget for the x data: x_select
x_select = Select(
    options=['col1, 'col2', 'col3', 'col4'],
    value='col1',
    title='x-axis data'
)

# Attach the update_plot callback to the 'value' property of x_select
x_select.on_change('value', update_plot)

# Create a dropdown Select widget for the y data: y_select
y_select = Select(
    options=['col1, 'col2', 'col3', 'col4'],
    value='col2',
    title='y-axis data'
)

# Attach the update_plot callback to the 'value' property of y_select
y_select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(widgetbox(slider, x_select, y_select), plot)
curdoc().add_root(layout)
```





# <a name="Data-Models"></a> Data Models
## Linear Models




# <a name="Linux-and-Bash-shells"></a> Linux and Bash shells
Google *Basic Bash Commands* for pwd, cp, cd, ls, cat, vim, nano, >, mv , sudo, apt update, apt upgrade, apt intall, etc...


## Login to a server
```
ssh -A yourAccountName@ipAddressOrHostName
```
Manage your ssh identity

activate your ssh-agent

```
eval $(ssh-agent -s)

```
add a ssh identity, for example your private key is of name id_rsa

```
ssh-add id_rsa
```


## change permission status of a file
```
chmod someCode yourFile
```
someCode=400 makes it non-writable by your own user. 
someCode=600 allows owner read-write not just read.

## Change Ownership of a file
```
chown new-owner  filename
```
new-owner: Specifies the user name or UID of the new owner of the file or directory.  
filename: Specifies the file or directory. 

## Compress/Decompress a file
Compress a file
```
tar cvzf filename.tar.gz filename
```

## File Transfer

Copy LocalFile to your remoteDestination
```
scp LocalFile RemoteDestinationFolder
```
Sometimes File Transfer may fail because perssion denied. You need to change ownership of the file.

## Activate and shut down your root power* (Caution, don't do this if you don't know why root power is dangerous.)
Activate: use `sudo su -`
Shut down: use `exit` or `logout`

## Check processes
```
ps aux
```

check processes with keyword

```
ps aux | grep agent
```

kill process

```
kill taskID
```
## Getting file from web on Linux Server
 First, install *wget* using  ` yum install wget` or `sudo apt-get install wget`.
 
 Then type:
 ```
wget yoururl
 ```

 One tip for getting url from a masked hyperlink like [this] on graphical user interface:
 right click the text and select 'Copy link address'.