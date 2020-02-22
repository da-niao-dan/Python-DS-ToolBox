
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