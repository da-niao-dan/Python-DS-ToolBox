# Pyspark ALS Model Applications
A lot of Notes here are from a nice Course in [DataCamp](https://campus.datacamp.com/courses/recommendation-engines-in-pyspark).


## Recommendation System with PySpark


The model traning process is similar to other pyspark model training. We could use pipelines.
### Toy 1

Prepare the Data
```python
# Import monotonically_increasing_id and show R
from pyspark.sql.functions import monotonically_increasing_id
R.show()

# Use the to_long() function to convert the dataframe to the "long" format.
ratings = to_long(R)
ratings.show()

# Get unique users and repartition to 1 partition
users = ratings.select("User").distinct().coalesce(1)

# Create a new column of unique integers called "userId" in the users dataframe.
users = users.withColumn("userId", monotonically_increasing_id()).persist()
users.show()

# Extract the distinct movie id's
movies = ratings.select("Movie").distinct()

# Repartition the data to have only one partition.
movies = movies.coalesce(1)

# Create a new column of movieId integers.
movies = movies.withColumn("movieID", monotonically_increasing_id()).persist()

# Join the ratings, users and movies dataframes
movie_ratings = ratings.join(users, "User", "left").join(movies, "Movie", "left")
movie_ratings.show()
```




Suppose we have pyspark dataframe called ratings:
```bash
In [1]: ratings.show(5)
+------+-------+------+
|userId|movieId|rating|
+------+-------+------+
|     2|      3|   3.0|
|     2|      1|   4.0|
|     2|      2|   4.0|
|     2|      0|   3.0|
|     0|      3|   4.0|
+------+-------+------+
only showing top 5 rows
```

```python
# Split the ratings dataframe into training and test data
(training_data, test_data) = ratings.randomSplit([0.8, 0.2], seed=42)

# Set the ALS hyperparameters
from pyspark.ml.recommendation import ALS
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank =10, maxIter =15, regParam =0.1,
          coldStartStrategy="drop", nonnegative =True, implicitPrefs = False)

# Fit the mdoel to the training_data
model = als.fit(training_data)

# Generate predictions on the test_data
test_predictions = model.transform(test_data)
test_predictions.show()

# Import RegressionEvaluator
from pyspark.ml.evaluation import RegressionEvaluator

# Complete the evaluator code
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# Extract the 3 parameters
print(evaluator.getMetricName())
print(evaluator.getLabelCol())
print(evaluator.getPredictionCol())

# Evaluate the "test_predictions" dataframe
RMSE = evaluator.evaluate(test_predictions)

# Print the RMSE
print (RMSE)
```

### Toy2
```python
# Look at the column names
print(ratings.columns)

# Look at the first few rows of data
print(ratings.show())
```

The output is like this:
```bash 
['userId', 'movieId', 'rating', 'timestamp']
    +------+-------+------+----------+
    |userId|movieId|rating| timestamp|
    +------+-------+------+----------+
    |     1|     31|   2.5|1260759144|
    |     1|   1029|   3.0|1260759179|
    |     1|   1061|   3.0|1260759182|
    |     1|   1129|   2.0|1260759185|
    |     1|   1172|   4.0|1260759205|
    |     1|   1263|   2.0|1260759151|
    |     1|   1287|   2.0|1260759187|
    |     1|   1293|   2.0|1260759148|
    |     1|   1339|   3.5|1260759125|
    |     1|   1343|   2.0|1260759131|
    |     1|   1371|   2.5|1260759135|
    |     1|   1405|   1.0|1260759203|
    |     1|   1953|   4.0|1260759191|
    |     1|   2105|   4.0|1260759139|
    |     1|   2150|   3.0|1260759194|
    |     1|   2193|   2.0|1260759198|
    |     1|   2294|   2.0|1260759108|
    |     1|   2455|   2.5|1260759113|
    |     1|   2968|   1.0|1260759200|
    |     1|   3671|   3.0|1260759117|
    +------+-------+------+----------+
    only showing top 20 rows
```

Calculate Sparsity:

```python
# Count the total number of ratings in the dataset
numerator = ratings.select("rating").count()

# Count the number of distinct userIds and distinct movieIds
num_users = ratings.select("userId").distinct().count()
num_movies = ratings.select("movieId").distinct().count()

# Set the denominator equal to the number of users multiplied by the number of movies
denominator = num_users * num_movies

# Divide the numerator by the denominator
sparsity = (1.0 - (numerator *1.0)/denominator)*100
print("The ratings dataframe is ", "%.2f" % sparsity + "% empty.")
```

```bash
The ratings dataframe is  98.36% empty.
```

Explore the dataset:
```python
# Import the requisite packages
from pyspark.sql.functions import col

# View the ratings dataset
ratings.show()

# Filter to show only userIds less than 100
ratings.filter(col("userId") < 100).show()

# Group data by userId, count ratings
ratings.groupBy("userId").count().show()

# Use .printSchema() to see the datatypes of the ratings dataset
ratings.printSchema()

# Tell Spark to convert the columns to the proper data types
ratings = ratings.select(ratings.userId.cast("integer"), ratings.movieId.cast("integer"), ratings.rating.cast("double"))

# Call .printSchema() again to confirm the columns are now in the correct format
ratings.printSchema()


```

Build the model:
```python
# Import the required functions
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create test and train set
(train, test) = ratings.randomSplit([0.80, 0.20], seed = 1234)

# Create ALS model
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative = True, implicitPrefs = False)

# Confirm that a model called "als" was created
type(als)

# Import the requisite items
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Add hyperparameters and their respective values to param_grid
param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [10, 50, 100, 150]) \
            .addGrid(als.maxIter, [5, 50, 100, 200]) \
            .addGrid(als.regParam, [.01, .05, .1, .15]) \
            .build()
           
# Define evaluator as RMSE and print length of evaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction") 
print ("Num models to be tested: ", len(param_grid))

# Build cross validation using CrossValidator
cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

# Confirm cv was built
print(cv)

# Print best_model
print(type(best_model))

# Complete the code below to extract the ALS model parameters
print("**Best Model**")

# Print "Rank"
print("  Rank:", best_model.getRank())

# Print "MaxIter"
print("  MaxIter:", best_model.getMaxIter())

# Print "RegParam"
print("  RegParam:", best_model.getRegParam())

# View the predictions 
test_predictions.show()

# Calculate and print the RMSE of test_predictions
RMSE = evaluator.evaluate(test_predictions)
print(RMSE)

# Look at user 60's ratings
print("User 60's Ratings:")
original_ratings.filter(col("userId") == 60).sort("rating", ascending = False).show()

# Look at the movies recommended to user 60
print("User 60s Recommendations:")
recommendations.filter(col("userId") == 60).show()

# Look at user 63's ratings
print("User 63's Ratings:")
original_ratings.filter(col("userId") == 63).sort("rating", ascending = False).show()

# Look at the movies recommended to user 63
print("User 63's Recommendations:")
recommendations.filter(col("userId") == 63).show()

```

### Implicit Ratings Model

Now we consider the dataset if we don't have explicit ratings.
Million Songs dataset, short hand:msd.
Look at the dataset:
```python
# Look at the data
msd.show()

# Count the number of distinct userIds
user_count = msd.select("userId").distinct().count()
print("Number of users: ", user_count)

# Count the number of distinct songIds
song_count = msd.select("songId").distinct().count()
print("Number of songs: ", song_count)
```

```bash
+------+------+---------+
    |userId|songId|num_plays|
    +------+------+---------+
    |   148|   148|        0|
    |   243|   496|        0|
    |    31|   471|        0|
    |   137|   463|        0|
    |   251|   623|        0|
    |    85|   392|        0|
    |    65|   540|        0|
    |   255|   243|        0|
    |    53|   516|        0|
    +------+------+---------+
    only showing top 10 rows
    
    Number of users:  321
    Number of songs:  729
```

```python
# Min num implicit ratings for a song
print("Minimum implicit ratings for a song: ")
msd.filter(col("num_plays") > 0).groupBy("songId").count().select(min("count")).show()

# Avg num implicit ratings per songs
print("Average implicit ratings per song: ")
msd.filter(col("num_plays") > 0).groupBy("songId").count().select(avg("count")).show()

# Min num implicit ratings from a user
print("Minimum implicit ratings from a user: ")
msd.filter(col("num_plays") > 0).groupBy("userId").count().select(min("count")).show()

# Avg num implicit ratings for users
print("Average implicit ratings per user: ")
msd.filter(col("num_plays") > 0).groupBy("userId").count().select(avg("count")).show()
```

```bash
<script.py> output:
    Minimum implicit ratings for a song: 
    +----------+
    |min(count)|
    +----------+
    |         3|
    +----------+
    
    Average implicit ratings per song: 
    +------------------+
    |        avg(count)|
    +------------------+
    |35.251063829787235|
    +------------------+
    
    Minimum implicit ratings from a user: 
    +----------+
    |min(count)|
    +----------+
    |        21|
    +----------+
    
    Average implicit ratings per user: 
    +-----------------+
    |       avg(count)|
    +-----------------+
    |77.42056074766356|
    +-----------------+
```

Fill missing values with 0:
```python
# View the data
Z.show()

# Extract distinct userIds and productIds
users = Z.select("userId").distinct()
products = Z.select("productId").distinct()

# Cross join users and products
cj = users.crossJoin(products)

# Join cj and Z
Z_expanded = cj.join(Z, ["userId", "productId"], "left").fillna(0)

# View Z_expanded
Z_expanded.show()
```

Tune Hyperparameters:

```python
ranks = [10, 20, 30, 40]
maxIters = [10, 20, 30, 40]
regParams = [.05, .1, .15]
alphas = [20, 40, 60, 80]

# For loop will automatically create and store ALS models
for r in ranks:
    for mi in maxIters:
        for rp in regParams:
            for a in alphas:
                model_list.append(ALS(userCol= "userId", itemCol= "songId", ratingCol= "num_plays", rank = r, maxIter = mi, regParam = rp, alpha = a, coldStartStrategy="drop", nonnegative = True, implicitPrefs = True))

# Print the model list, and the length of model_list
print (model_list, "Length of model_list: ", len(model_list))

# Validate
len(model_list) == (len(ranks)*len(maxIters)*len(regParams)*len(alphas))

```

Cross Validations:
```python
# Split the data into training and test sets
(training, test) = msd.randomSplit([0.8, 0.2])

#Building 5 folds within the training set.
train1, train2, train3, train4, train5 = training.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed = 1)
fold1 = train2.union(train3).union(train4).union(train5)
fold2 = train3.union(train4).union(train5).union(train1)
fold3 = train4.union(train5).union(train1).union(train2)
fold4 = train5.union(train1).union(train2).union(train3)
fold5 = train1.union(train2).union(train3).union(train4)

foldlist = [(fold1, train1), (fold2, train2), (fold3, train3), (fold4, train4), (fold5, train5)]

# Empty list to fill with ROEMs from each model
ROEMS = []

# Loops through all models and all folds
for model in model_list:
    for ft_pair in foldlist:

        # Fits model to fold within training data
        fitted_model = model.fit(ft_pair[0])

        # Generates predictions using fitted_model on respective CV test data
        predictions = fitted_model.transform(ft_pair[1])

        # Generates and prints a ROEM metric CV test data
        r = ROEM(predictions)
        print ("ROEM: ", r)

    # Fits model to all of training data and generates preds for test data
    v_fitted_model = model.fit(training)
    v_predictions = v_fitted_model.transform(test)
    v_ROEM = ROEM(v_predictions)

    # Adds validation ROEM to ROEM list
    ROEMS.append(v_ROEM)
    print ("Validation ROEM: ", v_ROEM)

# Import numpy
import numpy

# Find the index of the smallest ROEM
i = numpy.argmin(ROEMS)
print("Index of smallest ROEM:", i)

# Find ith element of ROEMS
print("Smallest ROEM: ", ROEMS[i])

# Extract the best_model
best_model = model_list[38]

# Extract the Rank
print ("Rank: ", best_model.getRank())

# Extract the MaxIter value
print ("MaxIter: ", best_model.getMaxIter())

# Extract the RegParam value
print ("RegParam: ", best_model.getRegParam())

# Extract the Alpha value
print ("Alpha: ", best_model.getAlpha())
```

Binary Ratings can use Implicit Ratings as well. In addition we could tweak the weights of users or movies (in ROEM)





### Other Resources

[Collaborative Filtering for Implitcit Feedback Datasets by Hu, Koren, Volinsky](http://yifanhu.net/PUB/cf.pdf)


[McKinsey&Company: "How Retailers Can Keep Up With Consumers"](https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers)

[ALS Data Preparation: Wide to Long Function](https://github.com/jamenlong/ALS_expected_percent_rank_cv/blob/master/wide_to_long_function.py)

[Hu, Koren, Volinsky: "Collaborative Filtering for Implicit Feedback Datasets"](http://yifanhu.net/PUB/cf.pdf)

[GitHub Repo: Cross Validation With Implicit Ratings in Pyspark](https://github.com/jamenlong/ALS_expected_percent_rank_cv/blob/master/ROEM_cv.py)

[Pan, Zhou, Cao, Liu, Lukose, Scholz, Yang: "One Class Collaborative Filtering"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.4684&rep=rep1&type=pdf)
