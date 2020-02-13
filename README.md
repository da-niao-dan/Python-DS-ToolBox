# Python-DS-ToolBox

Python Data Science Toolbox

## What is this note book about

This is my personal open notebook for Data Science in Python.
Unlike many books, this notebook is quite superficial. It covers major steps of doing simple data analysis and a lot of simple code examples.
I learnt the those tools from many sources, welcome to browse and edit this notebook.
This notebook is just a **remainder for what is available out there**. It does not offer any mathematical or technical or business details.

You can visit this [handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) for technical details.

## Quick advice

There are many great sources to learn Data Science, and here are some advice to dive into this field quickly:

1. Get some basic foundation in statistics and python. See www.hackerrank.com.
2. Get to know the general idea and areas of knowledge about data science.
3. Practice as you go. Google or Bing any term or question you don't understand. Stackoverflow and supporting documents for specific packages and functions are your best friend. During this process, do not lose sight of the big picture of data science.
4. Become a better data scientiest by doing more projects! (Don't try to memorize these tools, just do data science!)
5. Once comfortable with all materials in this notebook and engage in data analysis in business, you will know what skills to pick up next.

## Materials in this notebook

1. [Environment Configuration](#Environment-Configuration)
2. [Data Processing](#Data-Processing)
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
    * Dimension Reduction

3. [Exploring Data](#Exploring-Data)
    * Simple Data Visualization
    * Simple Statistical Tools

4. [Communicating with Data](#Communicating-with-Data)
    * Visualizing High-Dimension Data
    * Interative Data Visualization

5. [Data Models](#Data-Models)
    * Use Sklearn pipelines
    * Unsupervised Learning
    * Supervised Learning

6. [Git](#Git)
7. [Linux and Bash shells](#Linux-and-Bash-shells)
8. [Network Analysis](#Network-Analysis)
9. [PySpark](#PySpark)
10. Others
    * Efficient Coding in Python
    * Data Structure and Algorithms
    * Parallel and Cloud Computing with Spark technologies

---

## <a name="Environment-Configuration"></a> Environment Configuration

Choice of Working Environments: I recommend using *VScode* with remote *WSL* and *ssh on linux servers* for projects, while using *JupyterLab* for prototyping.

Manage of python packages: I recommend *Anaconda* for this task. 

Manage of version control and collaboration: Use *git* for personal projects, Use *Github* for open projects, *GitLab* for industrial projects.

Google or bing these keywords to find revelvant tutorials. This will be the last reminder of such actions.

### Prototyping Environment: conda environment quick set up with python jupyter notebook

Save environment to a text ﬁle.

install for example, jupyterlab.

```bash
conda install -c conda-forge jupyterlab
```

```bash
conda list --explicit > bio-env.txt
```

Create environment from a text ﬁle.

```bash
conda env create --file bio-env.txt nameHere
```

### Local Developing Environment setup: VScode with WSL

WSL is shorthand for Windows Subsystem for Linux, essentially, you can use linux working environment on Windows with ease. Install WSL according to this [guide](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

Then you start to work with VScode and WSL following this [guide](https://code.visualstudio.com/remote-tutorials/wsl/getting-started).

Remember to set up python3 environment according to this [guide](https://code.visualstudio.com/docs/languages/python). And the links in the guide above. Notice that, anaconda, jupyter notebook are all supported in VScode as well.

### Install anaconda on WSL inside VScode

Open bash terminal:

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
sha256sum filename  ### check the file is secure
bash Anaconda3--2019.10-Linux-x86_64.sh ## install it

conda update conda ## update conda to the newest version
```

### Getting access to Remote destinations: ssh

Get to know what is a [ssh](https://www.ssh.com/ssh/key) key.
Put it simply, you generate public and private key on your computer, you give ssh public key on remote machines which you have access for. Then you can log in to these remote destinations securely.

Following this [guide](https://help.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account) to add a ssh public key to your github account.

On Mac VScode bash, when you try to connect to a host with alias instead of IP, you may run into the problem of not able to connect to the remote. Now you need to edit host file on Mac and /etc/hosts in WSL to add IP address and Domain Name.

### Work on remote destinations using VScode

Follow this [guide](https://code.visualstudio.com/docs/remote/remote-overview).


### Alternative Developing Environment on Vim

This is for advanced user only. Ignore this if you are not comfortable with vim and git yet.

My current choice of working environment for python is vim.
Check this [guide](https://realpython.com/vim-and-python-a-match-made-in-heaven/). For the installation vim part, follow this [guide](https://github.com/ycm-core/YouCompleteMe/wiki/Building-Vim-from-source).




## <a name="Data-Processing"></a> Data Processing

### Getting Data

#### Read Data From Files

```python
import pandas as pd
```

##### csv file

csv stands for comma-separated values. Our target is to import csv file into a PandasDataFrame.
The first step is to understands what is the delimiter of the file.
Then we use the function read_csv in pandas.

Suppose f is a csv file with tab as delimiters, then we do this:

```python
df = pd.read_csv(f,delimiter ='\t')
```

##### json files

json files are human readable, you can think of them as python dictionaries. We want to load them into pandas DataFrame.
The furst step is to view your data to get a sense.
The second step is to use a function called json_normalize.

```python
from pandas.io.json import json_normalize

df = json_normalize(yourjsonfile,['key1', 'key2', ...], max_level=3)
```

#### Get data from APIs

Before we begin, have some basic knowledge on how the internet works and what is http, and what is API.

Getting data from web API usually require reading API manuals for specific APIs.

I recommend using package requests for this task in Python.

```python
# Assign URL to variable: url

url = 'yoururl' # Consult your api documents here.

header = {"Accept-Encoding":"gzip"} # Consult your api documents and requests package website here.

# Package the request, send the request and catch the response: r
param_dict = {your_parameters}# Consult your api documents and requests package website here.

with requests.Session() as s:
    download = s.post(url, data=param_dict,headers=header) ## consult post, get, and other http requests.
```

#### Get data from a SQL database

Here I will use a MySQL database for example. We are going to get data from MySQL database using SQL commands in python.

There are abundant choices of packages here. I use mysql-connector package.

```python
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

```sql
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

Use INTO statements to save query results into a table:

```sql
SELECT country_code, size,
  CASE WHEN size > 50000000
            THEN 'large'
       WHEN size > 1000000
            THEN 'medium'
       ELSE 'small' END
       AS popsize_group
INTO pop_plus
FROM populations
WHERE year = 2015;

-- 5. Select fields
select *
-- 1. From countries_plus (alias as c)
from countries_plus as c
  -- 2. Join to pop_plus (alias as p)
  inner join pop_plus as p
    -- 3. Match on country code
    on c.code = p.country_code
-- 4. Order the table
order by geosize_group asc;
```

### Organizing Data

#### Take a First Look

Before you do anything to your newly gained data, it's always a good idea to look at it first.

```python

import pandas as pd

## Suppose the data is loaded in a Pandas DataFrame data

## Look at the table as a whole:
data.info()
data.head()
data.describe()
```

Deal with categorical data:

First, save the names of non-numerical data in a list: LAEBLS.

Then, Convert them into suitable types.

Finally, take a look at unique values of each type.

```python

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
df[LABELS] = df[LABELS].apply(categorize_label,axis=0)

# Print the converted dtypes
print(df[LABELS].dtypes)


# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Calculate number of unique values for each label: num_unique_labels
num_unique_labels = df[LABELS].apply(pd.Series.nunique)

# Plot number of unique values for each label
num_unique_labels.plot(kind='bar')

# Label the axes
plt.xlabel('Labels')
plt.ylabel('Number of unique values')

# Display the plot
plt.show()

```

#### Data Cleaning

Basic str manipulations are quite handy in data cleaning.

##### Processing Datetimes

```python
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

##### Processing Lists

```python
### This document introduces some common operations of lists in Python

## Concatenate list of lists:
## Suppose x is an example list
x=[[1],[2],[3]]
## If we want [1,2,3] we can do:
new_list=[x[i][j] for i in range(len(x)) for j in range(len(x[i]))]
```

#### Transform DataFrames

```python
import pandas as pd

## Suppose df is a loaded DataFrame, 'column_name' is the name of one of its columns.

## Set a column as index column

df.set_index('column_name')


## suppose you have df1 and df2 and they both have a column named 'column_name'
## You can merge them using inner join. Save the merged DataFrame as df.
df = df1.merge(df2,how='inner',on='column_name')

### Deal with nans
df['DataFrame Column'] = df['DataFrame Column'].fillna(0)
df.fillna(0)
df.replace(np.nan,0)
```

### Dimension Reduction

#### Principal Component Analysis
PCA has the effects of decorrelating radom variables.

Take an example of grain samples with 2 measurements:length and width.

```python
# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)


# Show PCA on plot

# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()

## Show explained variable of PCA
plt.close()
# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
```

Use previous analysis to find intrinsic dimension of the dataset, then specify the number of dimensions to keep in the following PCA dimension reduction.
Finally, PCA for feature dimension reduction.

```python
# When the matrices are not sparse
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)

```
Let's consider a more detailed example: clustering documents

```python

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)

## Below, articles is a sparse matrix like csr_mat

# For sparse matrices, we have to use TruncatedSVD instead of PCA.
# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd,kmeans)

# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))
```

#### Non-negative matrix factorization

Interpretable Dimension Reduction for non-negative arrays.33

```python
# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler,nmf,normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)

# Import pandas
import pandas as pd

# Create a DataFrame: df
df = pd.DataFrame(norm_features,index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())
```


## <a name="Exploring-Data"></a> Exploring Data

### Simple Data Visualization

There are many tutorials for the famous *matplotlib* package for plotting in Python.
Therefore, here I will present an alternative package for plotting: *bokeh*.

I prefer Bokeh because it supports interactive design.

```python
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

### Simple Statistical Tools

#### For two continuous random variable

Calculate Pearson correlation coefficient.

```python
# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Assign the 0th column of grains: width
width = grains[:,0]

# Assign the 1st column of grains: length
length = grains[:,1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width,length)

# Display the correlation
print(correlation)
```

### For Categorical Data

Use Pearson's chi-sqaured test. Higher chi2 statistics, lower pval, indicate higher dependences.

```python
sklearn.feature_selection.chi2(X, y)
```

## <a name="Communicating-with-Data"></a> Communicating with Data

### Visualizing High-Dimension Data

#### t-SNE for 2D maps

t-SNE = "t-distributed stochastic neighbor embedding"

```python
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=200) ## try between 50 and 200

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs,ys,c=variety_numbers)
plt.show()
```

Another example:

```python
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs,ys,alpha=0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()
```

### Interactive Data Visualization

First, we make a plot.

```python
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

```python
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

```python
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

```python
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

```python
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

## <a name="Data-Models"></a> Data Models

### Use sklearn Pipelines

Example from a task to do supervised learning using both text and numeric data.

```python
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC, non-negative=True, norm=None, binary=False,
                                                   ngram_range=(1, 2))),  
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', PolynomialFeatures(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

```

### Unsupervised Learning

#### Clustering

##### K-means

Quick implementation

```python
# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

```

Fast Visualization

```python
# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys,c=labels,alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y,marker='D',s=50)
plt.show()

```

Choose then number of clusters by visualizing.

```python
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    # Fit model to samples
    model.fit(samples)
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
```

Exam Classification with crosstabulation if labels exist.

```python
# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['varieties'])

# Display ct
print(ct)


```

To optimize, try Standardize features and use pipelines.

##### Hierarchical clustering

###### agglomerative hierarchical clustering

Dendrograms plotting.

```python
# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# Calculate the linkage: mergings
mergings = linkage(normalize(samples),method='complete')

## Check out linkage methods: single, complete ...

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()

# Perform the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import fcluster

# Use fcluster to extract labels: labels
labels = fcluster(mergings,6,criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['varieties'])

# Display ct
print(ct)

```

### Supervised Learning

#### Ensemble Learning

Ensemble Learning is often used to combine models.

First, let's see the Hard Voting Classifier.

```python
# Set seed for reproducibility
SEED=1

# Instantiate lr
lr = LogisticRegression(random_state=SEED)

# Instantiate knn
knn = KNN(n_neighbors=27)

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]

# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:
    # Fit clf to the training set
    clf.fit(X_train, y_train)
    # Predict y_pred
    y_pred = clf.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))

# Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier

# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators=classifiers)

# Fit vc to the training set
vc.fit(X_train, y_train)

# Evaluate the test set predictions
y_pred = vc.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Voting Classifier: {:.3f}'.format(accuracy))

```

Then, let's see the Bagging method. Bagging=Bootstrap Aggregation.
Bootstrap method is not for combining different kinds of models. It trains using the *same algorithm* on different subsets of data.

```python

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier

# Instantiate dt
dt = DecisionTreeClassifier(max_depth=8,min_samples_leaf=8,random_state=1)

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, oob_score=True, random_state=1,n_jobs=-1)

# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate test set accuracy
acc_test = accuracy_score(y_test, y_pred)

# Evaluate OOB accuracy
acc_oob = bc.oob_score_

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))

```

Third, random forests. In Bagging, base estimators can be anything. Random forests use only decision trees. Similar to Bagging, each estimator is trained on a different bootstrap sample (same size as the training set). Unlike Bagging, RF introduces further randomization by sampling features at each node without replacement.

```python
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Instantiate rf
rf = RandomForestRegressor(n_estimators=25,
            random_state=2)

# Fit rf to the training set
rf.fit(X_train, y_train)

# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Predict the test set labels
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test,y_pred) ** (1/2)

# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()

```

Forth, Boosting. Many week predictors trained in sequence, each learn errors from before. Unlike RF, Boosting trees are in sequence not independent of each other.

AdaBoost.

```python

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

# Instantiate ada
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)

# Fit ada to the training set
ada.fit(X_train,y_train)

# Compute the probabilities of obtaining the positive class
y_pred_proba = ada.predict_proba(X_test)[:,1]

# Import roc_auc_score
from sklearn.metrics import roc_auc_score

# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))
```

Gradient Boosting. Like adaBoost, estimators are trained in sequence, each trying to correct the previous errors. While AdaBoost change weights of samples in training sequence, Gradient Boosting use weighted residuals during the sequence training process.

```python

# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate gb
gb = GradientBoostingRegressor(max_depth=4,
            n_estimators=200,
            random_state=2)

# Fit gb to the training set
gb.fit(X_train,y_train)

# Predict test set labels
y_pred = gb.predict(X_test)

# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute MSE
mse_test = MSE(y_test,y_pred)

# Compute RMSE
rmse_test = mse_test ** 0.5

# Print RMSE
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))

```

Stochastic Gradient Boosting. *Each tree is trained on a random subset of rows of training data.*

* The sampled instances(40%-80% of training set) are sampled without replacement. (Unlike Gradient Boosting)
* Features are sampled (without replacement) when choosing split points. (Like Gradient Boosting)
* Result: further ensemble diversity.
* Effect: adding further variance to the ensemble of trees.

```python
# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth=4,
            subsample=0.9,
            max_features=0.75,
            n_estimators=200,
            random_state=2)
# Fit sgbr to the training set
sgbr.fit(X_train,y_train)

# Predict test set labels
y_pred = sgbr.predict(X_test)

# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute test set MSE
mse_test = MSE(y_test,y_pred)

# Compute test set RMSE
rmse_test = mse_test ** 0.5

# Print rmse_test
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))
```

#### Hyperparameter Tuning

General Approaches:

* Grid Search
* Random Search
* Bayesian Optimization
* Genetic Algorithms
* ...

We first discuss grid search.

Grid Search:

* Manually set a grid of discrete hyperparameter values.
* Set a metric for scoring model performance
* Search exhaustively through the grid.
* For each set of hyperparameters, evaluate each model's CV score.
* The optimal hyperparameters are those of the model achieving the best CV score.

Let's tune a CART.

```python
# dt is a tree
# check hyper-parameters
dt.get_params
# Define params_dt
params_dt = {'max_depth':[2,3,4],
    'min_samples_leaf': [0.12,0.14,0.16,0.18]
}
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring='roc_auc',
                       cv=5,
                       n_jobs=-1)
# Import roc_auc_score from sklearn.metrics
from sklearn.metrics import roc_auc_score

# Extract the best estimator
best_model = grid_dt.best_estimator_

# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:,1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test,y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))
```

Let's tune a RF.

```python
# Check params
rf.get_params

# Define the dictionary 'params_rf'
params_rf = {'n_estimators':[100,350,500],
    'max_features':['log2','auto','sqrt'],
    'min_samples_leaf': [2,10,30]
}

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=3,
                       verbose=1,
                       n_jobs=-1)

grid_rf.fit(X_train,y_train)
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Extract the best estimator
best_model = grid_rf.best_estimator_

# Predict test set labels
y_pred = best_model.predict(X_test)

# Compute rmse_test
rmse_test = MSE(y_test,y_pred) ** (1/2)

# Print rmse_test
print('Test RMSE of best model: {:.3f}'.format(rmse_test))
```

#### Deep Learning

Keras short example.

```python
# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50,activation='relu',input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50,activation='relu'))

# Add the output layer
model.add(Dense(10,activation='softmax'))

# Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Fit the model
model.fit(X,y,validation_split=0.3)

```

# <a name="Git"> Git </a>

There is a official git [guide](https://git-scm.com/). Use it as reference and install git according to it.

However, here are some basic git commands to get you up to speed:

```bash
git init ###initialize a git repo locally at your current folder
git remote add origin linkToGitRepoOnline ## set a remote destination to sync with your git repo
git remote set-url origin linkToGitRepoOnline ## change the url of your remote called origin. from now on, origin stands for linkToGitRepoOnline
git pull origin branchNameHere ## pull data from remote branch to local repo

# Branching
git branch newBranch master ## create a new branch from master branch
git checkout -b branchNameHere ## switch to a branch
git branch -d branchNameHere ## delete a branch
git checkout master ## switch to master branch
git merge branchNameHere ## merge branchNameHere branch with current branch

# you modify or create some files
git add * ## stage all files you modified
git commit ## Say what you changed
git commit -a -m 'changes descriptions' ## quicker, but essentially same as above

git push origin master ## push changes to a remote branch master of origin, Here origin is a remote name, master is a branch name.
git push --set-upstream remoteNameHere branchNameHere ## If you run this once, later you only need:
git push
```

# <a name="Linux-and-Bash-shells">Linux and Bash shells</a> 

Google *Basic Bash Commands* for pwd, cp, cd, ls, cat, vim, nano, >, mv, sudo, apt update, apt upgrade, apt intall, man, du, df  etc...

## Login to a server

```bash
ssh -A yourAccountName@ipAddressOrHostName
```

## Manage your ssh identity

activate your ssh-agent

```bash
eval $(ssh-agent -s)

```

add a ssh identity, for example your private key is of name id_rsa

```bash
ssh-add id_rsa
```

## change permission status of a file

```bash
chmod someCode yourFile
```

someCode=400 makes it non-writable by your own user.

someCode=600 allows owner read-write not just read.

someCode=700 allows owner to read write and execute a file/folder.

**If you have problem with ssh logins, use the following command:**

```bash
chmod 700 -R ~/.ssh
chmod 400 ~/.ssh/id_rsa   ## assuming id_rsa is your private key
```

## Change Ownership of a file

```bash
chown new-owner  filename
```

new-owner: Specifies the user name or UID of the new owner of the file or directory.  
filename: Specifies the file or directory.

## File Transfer

Copy LocalFile to your remoteDestination

```bash
scp LocalFile RemoteDestinationFolder
```

Sometimes File Transfer may fail because perssion denied. You need to change ownership of the file.

## Activate and shut down your root power* (Caution, don't do this if you don't know why root power is dangerous.)

Activate: use `sudo su -`
Shut down: use `exit` or `logout`

## Check processes

check all processes

```bash
ps aux
```

check processes with keyword, for example: *agent*.

```bash
ps aux | grep agent

#Alternatively, directly get process ID
pgrep agent
```

kill process

```bash
kill taskID
```

## Getting file from web on Linux Server

First, install *wget* using  `yum install wget` or `sudo apt-get install wget`.

Then type:

```bash
wget yourUrl
```

One tip for getting url from a masked hyperlink like [this] on graphical user interface:
right click the text and select 'Copy link address'.

## compress and decompress files

### tar files

List the contents of a tar file:

```bash
tar -tvf archive.tar
```

Extract the contents of a tar file:

```bash
tar -xf archive.tar
```

Extract all the contents of a gzipped tar file:

```bash
tar -xzvf archive.tar.gz
```

Extract one file from a tar file:

```bash
tar -xvf archive.tar targetFileName
```

Extract some files specified by a particular format from a tar file:

```bash
tar -xvf archive.tar --wildcards '*2019.sh'
```

Create an tar Archive for a folder:

```bash
tar -cf archive.tar mydir/
```

Create an gzipped tar Archive for a folder :

```bash
tar -czf archive.tar.gz mydir/
```

### gzip files

decompress a file

```bash
gzip -d someFile.gz
```

compress a file
use option 1 to 9, 1 is for maximum compression at the slowest speed, 9 is for minimum compression at the fastest speed.

```bash
gzip -1 someFile
gzip -2 someFile
gzip -9 someFile
```

## Adding a PATH variable

Sometimes when you install a new software, you need to add a new PATH variable to your environment so that you can call the new software easily.

First you can run the following two lines:

```bash
PATH=YourPathToTheFolderOfTheNewSoftware:$PATH
export PATH
```

Then you can include these two lines in your ~/.bashrc file.

Sometimes you messed up the PATH variable by messed up ~/.bashrc file, (you find yourself cannot use ls or vim commands), relax, just run

```bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games
```

and debug ~/.bashrc file using vim.

```bash
vim ~/.bashrc
```

## Learn to use vim

Open bash and type:

```bash
vimtutor
```

## file editing

modify table using cut.

```bash

cut -d ' ' -f 1-10 orginal_filename > filename.
```

## crontab

### run python script in crontab

*VERY IMPORTANT!*

Assume you have multiple versions of python installed on your computer.
The first step is to locate your python executable file.
Now activate the environment you want to use.
Type:

```bash

which python
```

Copy and paste the path, and run python in crontab like this:

```bash
* * * * * nohup copiedPathToPython YourAbsolutePathToScript  >> /home/yourUserName/cron_lab.log 2>&1
```

### run python in crontab with characters other than English

For example, if you have Chinese character in your python script, you need to
include `# -*- coding: utf-8 -*-` at the top of python script.

## Set your sudo password feedback visible in *

```bash
sudo visudo
```

find the line with 'Defaults env_reset', change it to 'Defaults env_reset,pwfeedback'.

save the file.
Now refresh and test.

```bash
sudo -k
sudo ls
```

# <a name="Network-Analysis">Network Analysis</a> 

## Basic Analysis

import NetworkX API and draw a preloaded Network.

```python
# Import necessary modules
import matplotlib.pyplot as plt
import networkx as nx

# Draw the graph to screen
nx.draw(T_sub)
plt.show()
```

Get subsets of graph.

```python
# Use a list comprehension to get the nodes of interest: noi
noi = [n for n, d in T.nodes(data=True) if d['occupation'] == 'scientist']

# Use a list comprehension to get the edges of interest: eoi
eoi = [(u, v) for u, v, d in T.edges(data=True) if d['date'] < date(2010,1,1)]
```

There are many types of graph, such as directed, undirected, or multi graph.

Access edges of a graph and their metadata.

```python
# Set the weight of the edge
T.edges[1,10]['weight'] = 2

# Iterate over all the edges (with metadata)
for u, v, d in T.edges(data=True):

    # Check if node 293 is involved
    if 293 in [u,v]:

        # Set the weight to 1.1
        d['weight'] = 1.1
```

Check if a graph has a self-loop.

```python
# Define find_selfloop_nodes()
def find_selfloop_nodes(G):
    """
    Finds all nodes that have self-loops in the graph G.
    """
    nodes_in_selfloops = []

    # Iterate over all the edges of G
    for u, v in G.edges():

    # Check if node u and node v are the same
        if u==v:

            # Append node u to nodes_in_selfloops
            nodes_in_selfloops.append(u)

    return nodes_in_selfloops

# Check whether number of self loops equals the number of nodes in self loops
assert T.number_of_selfloops() == len(find_selfloop_nodes(T))
```

## Visualizing Networks

Three important types:

* Matrix plots
* Arc plots
* Circos plots

Matrix plot example:

```python
# Import nxviz
import nxviz as nv

# Create the MatrixPlot object: m
m = nv.MatrixPlot(T)

# Draw m to the screen
m.draw()

# Display the plot
plt.show()

# Convert T to a matrix format: A
A = nx.to_numpy_matrix(T)

# Convert A back to the NetworkX form as a directed graph: T_conv
T_conv = nx.from_numpy_matrix(A, create_using=nx.DiGraph())

# Check that the `category` metadata field is lost from each node
for n, d in T_conv.nodes(data=True):
    assert 'category' not in d.keys()
```

ArcPlot example:

```python
# Import necessary modules
import matplotlib.pyplot as plt
from nxviz import ArcPlot

# Create the un-customized ArcPlot object: a
a = ArcPlot(T)

# Draw a to the screen
a.draw()

# Display the plot
plt.show()

# Create the customized ArcPlot object: a2
a2 = ArcPlot(T,node_order='category',node_color='category')

# Draw a2 to the screen
a2.draw()

# Display the plot
plt.show()
```

Circos plot example:

```python
# Import necessary modules
import matplotlib.pyplot as plt
from nxviz import CircosPlot

# Create the CircosPlot object: c
c = CircosPlot(T)

# Draw c to the screen
c.draw()

# Display the plot
plt.show()
```

## Important Nodes

Degree centrality = n_neighbors/ possible_n_neighbors.

```python
# Define nodes_with_m_nbrs()
def nodes_with_m_nbrs(G,m):
    """
    Returns all nodes in graph G that have m neighbors.
    """
    nodes = set()

    # Iterate over all nodes in G
    for n in G.nodes():

        # Check if the number of neighbors of n matches m
        if len(list(G.neighbors(n))) == m:

            # Add the node n to the set
            nodes.add(n)

    # Return the nodes with m neighbors
    return nodes

# Compute and print all nodes in T that have 6 neighbors
six_nbrs = nodes_with_m_nbrs(T,6)
print(six_nbrs)

# Compute the degree of every node: degrees
degrees = [len(list(T.neighbors(n))) for n in T.nodes()]

# Print the degrees
print(degrees)

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Compute the degree centrality of the Twitter network: deg_cent
deg_cent = nx.degree_centrality(T)

# Plot a histogram of the degree centrality distribution of the graph.
plt.figure()
plt.hist(list(deg_cent.values()))
plt.show()

# Plot a histogram of the degree distribution of the graph
plt.figure()
plt.hist([len(list(T.neighbors(n))) for n in T.nodes()])
plt.show()

# Plot a scatter plot of the centrality distribution and the degree distribution
plt.figure()
plt.scatter([len(list(T.neighbors(n))) for n in T.nodes()], list(deg_cent.values()))
plt.show()
```

Path between nodes:

```python
def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()
    queue = [node1]

    for node in queue:
        neighbors = G.neighbors(node)
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break

        else:
            visited_nodes.add(node)
            queue.extend([n for n in neighbors if n not in visited_nodes])

        # Check to see if the final element of the queue has been reached
        if node == queue[-1]:
            print('Path does not exist between nodes {0} and {1}'.format(node1, node2))

            # Place the appropriate return statement
            return False
```

Betweenness centrality = n_shortest_paths_through_node/all_possible_shortest_paths.

```python
# Compute the betweenness centrality of T: bet_cen
bet_cen = nx.betweenness_centrality(T)

# Compute the degree centrality of T: deg_cen
deg_cen = nx.degree_centrality(T)

# Create a scatter plot of betweenness centrality and degree centrality
plt.scatter(list(bet_cen.values()),list(deg_cen.values()))

# Display the plot
plt.show()

# Define find_nodes_with_highest_deg_cent()
def find_nodes_with_highest_deg_cent(G):

    # Compute the degree centrality of G: deg_cent
    deg_cent = nx.degree_centrality(G)

    # Compute the maximum degree centrality: max_dc
    max_dc = max(list(deg_cent.values()))

    nodes = set()

    # Iterate over the degree centrality dictionary
    for k, v in deg_cent.items():

        # Check if the current value has the maximum degree centrality
        if v == max_dc:

            # Add the current node to the set of nodes
            nodes.add(k)

    return nodes

# Find the node(s) that has the highest degree centrality in T: top_dc
top_dc = find_nodes_with_highest_deg_cent(T)
print(top_dc)

# Write the assertion statement
for node in top_dc:
    assert nx.degree_centrality(T)[node] == max(nx.degree_centrality(T).values())

# Define find_node_with_highest_bet_cent()
def find_node_with_highest_bet_cent(G):

    # Compute betweenness centrality: bet_cent
    bet_cent = nx.betweenness_centrality(G)

    # Compute maximum betweenness centrality: max_bc
    max_bc = max(list(bet_cent.values()))

    nodes = set()

    # Iterate over the betweenness centrality dictionary
    for k, v in bet_cent.items():

        # Check if the current value has the maximum betweenness centrality
        if v == max_bc:

            # Add the current node to the set of nodes
            nodes.add(k)

    return nodes

# Use that function to find the node(s) that has the highest betweenness centrality in the network: top_bc
top_bc = find_node_with_highest_bet_cent(T)
print(top_bc)

# Write an assertion statement that checks that the node(s) is/are correctly identified.
for node in top_bc:
    assert nx.betweenness_centrality(T)[node] == max(nx.betweenness_centrality(T).values())
```

## Structures

Cliques: fully connected subgraph.

Identyfying triangle relationships.

```python
from itertools import combinations

# Define is_in_triangle()
def is_in_triangle(G, n):
    """
    Checks whether a node `n` in graph `G` is in a triangle relationship or not.

    Returns a boolean.
    """
    in_triangle = False

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n),2):

        # Check if an edge exists between n1 and n2
        if G.has_edge(n1,n2):
            in_triangle = True
            break
    return in_triangle

from itertools import combinations

# Write a function that identifies all nodes in a triangle relationship with a given node.
def nodes_in_triangle(G, n):
    """
    Returns the nodes in a graph `G` that are involved in a triangle relationship with the node `n`.
    """
    triangle_nodes = set([n])

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n),2):

        # Check if n1 and n2 have an edge between them
        if G.has_edge(n1,n2):

            # Add n1 to triangle_nodes
            triangle_nodes.add(n1)

            # Add n2 to triangle_nodes
            triangle_nodes.add(n2)

    return triangle_nodes

# Write the assertion statement
assert len(nodes_in_triangle(T,1)) == 35

from itertools import combinations

# Define node_in_open_triangle()
def node_in_open_triangle(G, n):
    """
    Checks whether pairs of neighbors of node `n` in graph `G` are in an 'open triangle' relationship with node `n`.
    """
    in_open_triangle = False

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n),2):

        # Check if n1 and n2 do NOT have an edge between them
        if not G.has_edge(n1,n2):

            in_open_triangle = True

            break

    return in_open_triangle

# Compute the number of open triangles in T
num_open_triangles = 0

# Iterate over all the nodes in T
for n in T.nodes():

    # Check if the current node is in an open triangle
    if node_in_open_triangle(T, n):

        # Increment num_open_triangles
        num_open_triangles += 1

print(num_open_triangles)

# Define maximal_cliques()
def maximal_cliques(G,size):
    """
    Finds all maximal cliques in graph `G` that are of size `size`.
    """
    mcs = []
    for clique in nx.find_cliques(G):
        if len(clique) == size:
            mcs.append(clique)
    return mcs

# Check that there are 33 maximal cliques of size 3 in the graph T
assert len(maximal_cliques(T,3)) == 33
```

Subgraphs.

```python
nodes_of_interest = [29, 38, 42]

# Define get_nodes_and_nbrs()
def get_nodes_and_nbrs(G, nodes_of_interest):
    """
    Returns a subgraph of the graph `G` with only the `nodes_of_interest` and their neighbors.
    """
    nodes_to_draw = []

    # Iterate over the nodes of interest
    for n in nodes_of_interest:

        # Append the nodes of interest to nodes_to_draw
        nodes_to_draw.append(n)

        # Iterate over all the neighbors of node n
        for nbr in G.neighbors(n):

            # Append the neighbors of n to nodes_to_draw
            nodes_to_draw.append(nbr)

    return G.subgraph(nodes_to_draw)

# Extract the subgraph with the nodes of interest: T_draw
T_draw = get_nodes_and_nbrs(T,nodes_of_interest)

# Draw the subgraph to the screen
nx.draw(T_draw)
plt.show()

# Extract the nodes of interest: nodes
nodes = [n for n, d in T.nodes(data=True) if d['occupation'] == 'celebrity']

# Create the set of nodes: nodeset
nodeset = set(nodes)

# Iterate over nodes
for n in nodes:

    # Compute the neighbors of n: nbrs
    nbrs = T.neighbors(n)

    # Compute the union of nodeset and nbrs: nodeset
    nodeset = nodeset.union(nbrs)

# Compute the subgraph using nodeset: T_sub
T_sub = T.subgraph(nodeset)

# Draw T_sub to the screen
nx.draw(T_sub)
plt.show()

```

# <a name="PySpark">PySpark</a> 

When to work with Spark?
Normally we just use pandas. However, pandas gets extremely slow when the datasets becomes larger.

Before jumping to Spark, be clear about your purpose. If you just want to process several files with ~2GB size, and that they can be processed line by line without further calculations, you may want to go for line by line approach, because pandas becomes slow when Input-Output flow is large.

If you need to perform more calculations or the size of tables gets to TB size or larger size, you may want to consider to use Spark!

## First Impression of Spark

Spark is a technology for parellel computing on clusters. I would like to think of it as pandas on clusters (very inaccurate analogy).
Resilient Distributed Dataset is the basic building blocks in Spark. DataFrame is built upon RDD with built in optimizations when it comes to table operations. I would like to think of it as DataFame in pandas (again, just an analogy).

PySpark reference is [here](http://spark.apache.org/docs/2.1.0/api/python/pyspark.html).

To use spark in Python, you need to instantiate a SparkContext object in python. You can think of it as a connection to your control server of your cluster.
Then you need to create a SparkSession, of which you can think as an interface to this connection.

```python
## Assume we have loaded a SparkSession called spark
## Here we are going to practice how to create a SparkSession
# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession

# Create my_spark
my_spark = SparkSession.builder.getOrCreate()

# Print
print(my_spark)
```

From now on let's assume spark stands for SparkSession NOT A SparkContext anymore. Just a naming stuff.
Your SparkSession has an attribute called catalog with lists of all data inside the session on your cluster. There are several methods to get information.
For example:

```python
# Print the tables in the catalog
print(spark.catalog.listTables())
```

Amazingly, you can quey tables in spark sessions like datasets in sql databases.

```python
# Assume a table of flights is shown in your catalog
query = "FROM flights SELECT * LIMIT 10"

# Get the first 10 rows of flights, flights10 will be a DataFrame
flights10 = spark.sql(query)

# Show the results
flights10.show()
```

You can convert spark DataFrames to pandas DataFrames and work on it locally:

```python
# Example query
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"

# Run the query
flight_counts = spark.sql(query)

# Convert the results to a pandas DataFrame
pd_counts = flight_counts.toPandas()

# Print the head of pd_counts
print(pd_counts.head())
```

Convert a pandas DataFrame to a spark DataFrame and (not automatically but after some action) work on clusters!

```python
# Create pd_temp
pd_temp = pd.DataFrame(np.random.random(10))

# Create spark_temp from pd_temp
spark_temp = spark.createDataFrame(pd_temp)

# Examine the tables in the catalog
print(spark.catalog.listTables())

# Add spark_temp to the catalog, register it using name "temp", createOrReplaceTempView garantees your table names are not duplicates, it will update existing table if exists.
spark_temp.createOrReplaceTempView("temp")

# Examine the tables in the catalog again
print(spark.catalog.listTables())
```

Without working with pandas, let's load the data directly.

```python
# Take an example
file_path = "/usr/local/share/datasets/airports.csv"

# Read in the airports data
airports = spark.read.csv(file_path,header=True)

# Show the data
airports.show()
```

## Manipulating Data

From now on, we have to recognize the unique functionalities of Spark. Forget about pandas dataframes, because those intuitions are not helpful anymore.

Column is an Object type in Spark, and it can be created by `df.colName`.

Unlike pandas DataFrame, Spark DataFrame is immutable, meaning you cannot change columns in place.

To add new column generated by some operation on old column to a df you do something like this: `df=df.withColumn("newCol", df.oldCol + 1)`.
To replace an old column you do this:`df=df.withColumn("oldCol", df.oldCol + 1)`.

```python
# Create the DataFrame flights
flights = spark.table("flights")

# Show the head
flights.show()

# Add duration_hrs
flights = flights.withColumn("duration_hrs", flights.air_time/60.)

# Rename column A to B
flights = flights.withColumnRenamed("A","B")
```

### Filtering Data

Passing string of SQL code or Booleans are the same.

```python
# Filter flights by passing a string
long_flights1 = flights.filter("distance > 1000")

# Filter flights by passing a column of boolean values
long_flights2 = flights.filter(flights.distance > 1000)

# Print the data to check they're equal
long_flights1.show()
long_flights2.show()
```

You may consider filter is like 'SELECT * FROM table, WHERE(your SQL filtering condition here)'

### Selecting

Recall the 'SELECT' statement in SQL.
The 'select' in DataFrame is even more powerful.
'selectExpr' is basically equivalent to 'select' but taking string SQL code as argument.

```python
# Select the first set of columns
selected1 = flights.select('tailnum','origin','dest')

# Select the second set of columns
temp = flights.select(flights.origin, flights.dest, flights.carrier)

# Define first filter
filterA = flights.origin == "SEA"

# Define second filter
filterB = flights.dest == "PDX"

# Filter the data, first by filterA then by filterB
selected2 = temp.filter(filterA).filter(filterB)

# Define avg_speed
avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")

# Select the correct columns
speed1 = flights.select("origin", "dest", "tailnum", avg_speed)

# Create the same table using a SQL expression
speed2 = flights.selectExpr("origin", "dest", "tailnum", "distance/(air_time/60) as avg_speed")
```

### Aggregating

Aggregating means summerizing a group of data in some sense. like min(), max(), count().

The .groupBy() method of the DataFrame creates an object of type pyspark.sql.GroupedData. Passing arguments to .groupby() is similar to using groupby in SQL.

```python
# Find the shortest flight from PDX in terms of distance
flights.filter(flights.origin=='PDX').groupBy().min('distance').show()

# Find the longest flight from SEA in terms of air time
flights.filter(flights.origin=='SEA').groupBy().max('air_time').show()

# Average duration of Delta flights
flights.filter(flights.carrier=="DL").filter(flights.origin=="SEA").groupBy().avg('air_time').show()

# Total hours in the air
flights.withColumn("duration_hrs", flights.air_time/60).groupBy().sum("duration_hrs").show()

# Group by tailnum
by_plane = flights.groupBy("tailnum")

# Number of flights each plane made
by_plane.count().show()

# Group by origin
by_origin = flights.groupBy("origin")

# Average duration of flights from PDX and SEA
by_origin.avg("air_time").show()
```

GroupData objects have another useful method .agg(), which allows you to pass an aggregat column expression that uses any of the aggregate functions from the pyspark.sql.functions submodule.
This submodule has many useful functions.

Example:

```python
# Import pyspark.sql.functions as F
import pyspark.sql.functions as F

# Group by month and dest
by_month_dest = flights.groupBy('month','dest')

# Average departure delay by month and destination
by_month_dest.avg("dep_delay").show()

# Standard deviation of departure delay
by_month_dest.agg(F.stddev("dep_delay")).show()
```

### Joining

```python
# Examine the data
print(airports.show())

# Rename the faa column
airports = airports.withColumnRenamed("faa","dest")

# Join the DataFrames
flights_with_airports = flights.join(airports,'dest',how='leftouter')

# Examine the new DataFrame
print(flights_with_airports.show())
```

## Machine Learning Pipelines

Before we get to the pipeline, you should understand that there are two main classes for spark: Estimators and Transformers. Estimators have method .fit() and return models. Transformers have method .transform and return DataFrames. Spark modelling mainly relies on these two classes.

First, preprocessing data. Spark requires numerical data for modeling.

```python
# Rename year column
planes = planes.withColumnRenamed("year", "plane_year")

# Join the DataFrames
model_data = flights.join(planes, on='tailnum', how="leftouter")

# Cast the columns to integers
model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast('integer'))
model_data = model_data.withColumn("air_time", model_data.air_time.cast('integer'))
model_data = model_data.withColumn("month", model_data.month.cast('integer'))
model_data = model_data.withColumn("plane_year", model_data.plane_year.cast('integer'))

# Create the column plane_age
model_data = model_data.withColumn("plane_age", model_data.year - model_data.plane_year)

#Create is_late
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)

# Convert to an integer
model_data = model_data.withColumn("label", model_data.is_late.cast('integer'))

# Remove missing values
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")
```

Next, process categorical data.

```python
# Create a StringIndexer
carr_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_index")

# Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol="carrier_index", outputCol="carrier_fact")

# Create a StringIndexer
dest_indexer = StringIndexer(inputCol="dest", outputCol="dest_index")

# Create a OneHotEncoder
dest_encoder = OneHotEncoder(inputCol="dest_index", outputCol="dest_fact")
```

The next step is to combine all of the columns of features to a single column.

```python
# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")
```

Next, use Pipeline to combine all Transformers and Estimators.

```python
# Import Pipeline
from pyspark.ml import Pipeline

# Make the pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])
```

Then we are going to split the data, *after* these transformations. Operations like StringIndexer don't always give the same index even with the same list of strings.

```python
# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)

# Split the data into training and test sets
training, test = piped_data.randomSplit([.6, .4])
```

OK, finally the fun part!

```python
# Import LogisticRegression
from pyspark.ml.classification import LogisticRegression

# Create a LogisticRegression Estimator
lr = LogisticRegression()

# Import the evaluation submodule
import pyspark.ml.evaluation as evals

# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")

# Import the tuning submodule
import pyspark.ml.tuning as tune

# Create the parameter grid
grid = tune.ParamGridBuilder()

# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01)) ## model
grid = grid.addGrid(lr.elasticNetParam, [0 , 1]) ## regularization

# Build the grid
grid = grid.build()

# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator
               )

# Fit cross validation models
models = cv.fit(training)

# Extract the best model
best_lr = models.bestModel

# Use the model to predict the test set
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))
```

### Details about Creating a SparkSession

```python
# Import the PySpark module
from pyspark.sql import SparkSession

# Create SparkSession object
spark = SparkSession.builder \
                    .master('local[*]') \ ## To connect to a remote location, use: spark://<IP address | DNS name>:<port>
                    .appName('test') \
                    .getOrCreate()

# What version of Spark?
print(spark.version)

# Terminate the cluster
spark.stop()
```

### Details about loading data into Spark

DataFrame:
Select Methods:

* count()
* show()
* printSchema()

Selected attributes:

* dtypes

Reading data from csv: `cars = spark.read.csv("cars.csv",header=True)
Optional arguments:

* header
* sep
* schema - explicit column data types
* inferSchema - deduce column data types?
* nullValue -placeholder for missing data

This action can have problems. We may prefer `cars = spark.read.csv("cars.csv", header=True, inferSchema=True, nullValue='NA')` (Always good to explicitly define missing values.)

check `cars.dtypes` to show datatypes of each column. It turns out, columns with missing data will have 'NA' string, that column will be wrongly interpretated as String column.

In that case, we need to specify the schema by hand.

```python
# Read data from CSV file
flights = spark.read.csv('flights.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

# Get number of records
print("The data contain %d records." % flights.count())

# View the first five records
flights.show(5)

# Check column data types
flights.dtypes

from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Specify column names and types
schema = StructType([
    StructField("id", IntegerType()),
    StructField("text", StringType()),
    StructField("label", IntegerType())
])

# Load data from a delimited file
sms = spark.read.csv('sms.csv', sep=';', header=False, schema=schema)

# Print schema of DataFrame
sms.printSchema()
```

### Details about manipulating data

Data selection

```python

# Either drop the columns you don't want
cars = cars.drop('maker','model')

# ... or select the columns you do want
cars = cars.select("origin", 'type', 'cyl')

# Filtering out missing vals
## count
cars.filter('cyl is NULL').count()

## drop records with missing values in the cylinders column
cars = cars.filter('cyl IS NOT NULL')

## drop records with any missing data
cars = cars.dropna()
```

Index categorical data.

```python
from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol='type',outputCol='type_idx')

# Assign index values to strings
indexer = StringIndexer(inputCol= 'type',
                        outputCol='type_idx')

# Assign index values to strings
indexer = indexer.fit(cars)

# Create column with index values
cars = indexer.transform(cars)
```

By defaults the most frequent string will get index 0, and the  least frequent string will get the maximum index.

Use `stringOrderType` to change order.

Assembling cobv ][-=]
