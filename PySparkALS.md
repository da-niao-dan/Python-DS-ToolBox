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









### Other Resources

[McKinsey&Company: "How Retailers Can Keep Up With Consumers"](https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers)

[ALS Data Preparation: Wide to Long Function](https://github.com/jamenlong/ALS_expected_percent_rank_cv/blob/master/wide_to_long_function.py)

[Hu, Koren, Volinsky: "Collaborative Filtering for Implicit Feedback Datasets"](http://yifanhu.net/PUB/cf.pdf)

[GitHub Repo: Cross Validation With Implicit Ratings in Pyspark](https://github.com/jamenlong/ALS_expected_percent_rank_cv/blob/master/ROEM_cv.py)

[Pan, Zhou, Cao, Liu, Lukose, Scholz, Yang: "One Class Collaborative Filtering"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.4684&rep=rep1&type=pdf)
