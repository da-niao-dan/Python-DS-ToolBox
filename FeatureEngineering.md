# Feature Engineering
Yin-Yang sides of feature engineering.


## Transforming or creating new features (Yang-side)

We can combine tables, transform features to create new features.
We can even do so with help. Let's see [featuretools](https://www.featuretools.com/)

After we created "meaningful" features, we are now ready to transform their format.
### OneHot Encoding: Dealing with discrete features

### Mathematical Operations: Creating New Features

#### Single Column
Apply functions such as log, sqrt, pow, or other functions that take 1 input.

#### Two Columns
Apply product,ratio, or other transformations that take 2 or more inputs.

### Bucketing: Dealing with continuous feature

### NLP: dealing with text feature

## Dimension Reduction (Yin-side)

### Principal Component Analysis (PCA)

### auto-encoding using deep neural networks

### Feature selection

Before the final task, we could try to solve a representive task. Use feature importance for a model, usually trees, to select features. Use SelectKBest (e.g. [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)).
In addition, we can do selection while training - LASSO regularization etc.


