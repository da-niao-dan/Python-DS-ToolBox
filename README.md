# Python-DS-ToolBox

Python Data Science Toolbox

## What is this note book about

This is my personal open notebook for Data Science in Python. Its target is people who have no or little experience in Data Science.
Unlike many books, this notebook is quite short and superficial. It covers major steps of doing simple data analysis.
These are simple examples for performing the tasks. Always consult documents, or google when using the functions if you have any doubt.
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

6. Other important skills
    * git
    * [Linux and Bash shells](#Linux-and-Bash-shells)
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


## Git

There is a official git [guide](https://git-scm.com/). Use it as reference and install git according to it.

However, here are some basic git commands to get you up to speed:

```bash
git init ###initialize a git repo locally at your current folder
git remote add origin linkToGitRepoOnline ## set a remote destination to sync with your git repo
git remote set-url origin linkToGitRepoOnline ## change the url of your remote called origin. from now on, origin stands for linkToGitRepoOnline
git pull origin branchNameHere ## pull data from remote branch to local repo

### Branching
git branch newBranch master ## create a new branch from master branch
git checkout -b branchNameHere ## switch to a branch
git branch -d branchNameHere ## delete a branch
git checkout master ## switch to master branch
git merge branchNameHere ## merge branchNameHere branch with current branch

#### you modify or create some files
git add * ## stage all files you modified
git commit ## Say what you changed
git commit -a -m 'changes descriptions' ## quicker, but essentially same as above

git push origin master ## push changes to a remote branch master of origin, Here origin is a remote name, master is a branch name.
git push --set-upstream remoteNameHere branchNameHere ## If you run this once, later you only need:
git push
```


## <a name="Linux-and-Bash-shells"></a> Linux and Bash shells

Google *Basic Bash Commands* for pwd, cp, cd, ls, cat, vim, nano, >, mv, sudo, apt update, apt upgrade, apt intall, man, du, df  etc...

### Login to a server

```bash
ssh -A yourAccountName@ipAddressOrHostName
```

### Manage your ssh identity

activate your ssh-agent

```bash
eval $(ssh-agent -s)

```

add a ssh identity, for example your private key is of name id_rsa

```bash
ssh-add id_rsa
```

### change permission status of a file

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

### Change Ownership of a file

```bash
chown new-owner  filename
```

new-owner: Specifies the user name or UID of the new owner of the file or directory.  
filename: Specifies the file or directory.

### File Transfer

Copy LocalFile to your remoteDestination

```bash
scp LocalFile RemoteDestinationFolder
```

Sometimes File Transfer may fail because perssion denied. You need to change ownership of the file.

### Activate and shut down your root power* (Caution, don't do this if you don't know why root power is dangerous.)

Activate: use `sudo su -`
Shut down: use `exit` or `logout`

### Check processes

check all processes

```bash
ps aux
```

check processes with keyword, for example: *agent*.

```bash
ps aux | grep agent
```

kill process

```bash
kill taskID
```

### Getting file from web on Linux Server

First, install *wget* using  `yum install wget` or `sudo apt-get install wget`.

Then type:

```bash
wget yourUrl
```

One tip for getting url from a masked hyperlink like [this] on graphical user interface:
right click the text and select 'Copy link address'.

### compress and decompress files

#### tar files

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

#### gzip files

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

### Adding a PATH variable

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

### Learn to use vim

Open bash and type:

```bash
vimtutor
```
### crontab

#### run python script in crontab

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

#### run python in crontab with characters other than English
For example, if you have Chinese character in your python script, you need to
include ` # -*- coding: utf-8 -*-` at the top of python script. 



### Set your sudo password feedback visible in *

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




