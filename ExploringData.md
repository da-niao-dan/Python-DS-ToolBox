

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
