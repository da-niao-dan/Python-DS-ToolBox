# Feature Engineering
Yin-Yang sides of feature engineering.


## Transforming or creating new features (Yang-side)

We can combine tables, transform features to create new features.
We can even do so with help. Let's see [featuretools](https://www.featuretools.com/)

After we created "meaningful" features, we are now ready to transform their format.


## Dimension Reduction (Yin-side)

There are many available tools for this.

### Statistics: remove highly correlated data
We can do this automatically using [featuretools](https://www.featuretools.com/).
Or we can remove them by hand.

### Principal Component Analysis (PCA)
This is a classic and fast method, but it has it's limitations.
Remember standardize your data before you do this.
We can do this using sklearn very fast.

### auto-encoding using deep neural networks
We need sufficient data to do this well, more complicated than PCA>


### Feature selection
Before the final task, we could try to solve a representive task. Use feature importance for a model, usually trees, to select features. Use SelectKBest (e.g. [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)).
In addition, we can do selection while training - LASSO regularization etc.


